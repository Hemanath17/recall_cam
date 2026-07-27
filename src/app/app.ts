import {
  Component,
  ElementRef,
  ViewChild,
  NgZone,
  OnDestroy,
  HostListener,
  signal,
  computed,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import {
  ObjectDetector,
  HandLandmarker,
  FilesetResolver,
} from '@mediapipe/tasks-vision';

export interface CamEvent {
  id: number;
  time: Date;
  action: string;
  object: string;
  person: string;
  confidence: number;
  thumbnail: string;
}

interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface Track {
  id: number;
  label: string;
  displayName: string;
  box: Box;
  lastSeenFrame: number;
  handNearFrame: number;
  restingCentroid: { x: number; y: number };
  stableFrames: number;
}

/** Tuning knobs. Change these first when behaviour is wrong. */
const CONFIG = {
  /** Detection confidence floor. Lower if your object is missed. */
  scoreThreshold: 0.4,
  /** Box overlap needed to call two detections the same object. */
  iouMatch: 0.3,
  /** Pixels. A hand counts as near an object below this distance. */
  handNear: 90,
  /** Frames an object must be missing before a pickup fires. */
  goneMin: 15,
  /** Frames after which a disappearance is too old to matter. */
  goneMax: 40,
  /** How recently a hand must have been near, in frames. */
  handRecent: 45,
  /** Frames before a lost track is forgotten entirely. */
  trackExpiry: 60,
  /** Pixels of centroid drift that still counts as stationary. */
  stillRadius: 15,
  /** Pixels of centroid drift that counts as a deliberate move. */
  moveRadius: 40,
  /** Frames an object must sit still before a move can register. */
  stableBeforeMove: 30,
  /** Frames between buffer snapshots. */
  bufferEvery: 5,
  /** Snapshots retained, used to find the "before" image. */
  bufferSize: 30,
  /** Set false to send real requests to /api/caption. */
  useFakeCaptions: true,
  /** Draw the live hand-distance readout. Turn off before demoing. */
  showTuning: true,
};

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class AppComponent implements OnDestroy {
  @ViewChild('video') videoRef!: ElementRef<HTMLVideoElement>;
  @ViewChild('canvas') canvasRef!: ElementRef<HTMLCanvasElement>;

  readonly running = signal(false);
  readonly modelsReady = signal(false);
  readonly statusText = signal('Camera off');
  readonly fps = signal(0);
  readonly events = signal<CamEvent[]>([]);
  readonly query = signal('');
  readonly cameraError = signal('');
  /** Nothing is logged until this is set. */
  readonly watchedObject = signal('');

  readonly filteredEvents = computed(() => {
    const q = this.query().trim().toLowerCase();
    const all = this.events();
    if (!q) return all;
    return all.filter((e) =>
      `${e.action} ${e.object} ${e.person}`.toLowerCase().includes(q)
    );
  });

  private stream: MediaStream | null = null;
  private rafId = 0;
  private nextId = 1;
  private frameTimes: number[] = [];

  private detector?: ObjectDetector;
  private handLandmarker?: HandLandmarker;

  private tracks = new Map<number, Track>();
  private nextTrackId = 1;
  private personCount = 0;
  private frameNo = 0;

  private handPoints: { x: number; y: number }[] = [];
  private frameBuffer: string[] = [];

  constructor(private zone: NgZone) {}

  trackById(_: number, e: CamEvent): number {
    return e.id;
  }

  /** Bind an input to this to choose what to watch. */
  setWatchedObject(value: string): void {
    this.watchedObject.set(value.trim().toLowerCase());
    this.tracks.clear();
    if (this.running()) this.updateStatus();
  }

  private updateStatus(): void {
    this.statusText.set(
      this.watchedObject()
        ? `Watching for ${this.watchedObject()}`
        : 'Running. Enter an object to watch.'
    );
  }

  // ---------------------------------------------------------------- models

  private async loadModels(): Promise<void> {
    if (this.modelsReady()) return;
    this.statusText.set('Loading models…');

    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
    );

    this.detector = await ObjectDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite',
        delegate: 'GPU',
      },
      scoreThreshold: CONFIG.scoreThreshold,
      runningMode: 'VIDEO',
    });

    this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        delegate: 'GPU',
      },
      numHands: 2,
      runningMode: 'VIDEO',
    });

    this.modelsReady.set(true);
  }

  // ---------------------------------------------------------------- camera

  async start(): Promise<void> {
    if (this.running()) return;

    this.cameraError.set('');
    this.statusText.set('Requesting camera…');

    if (!window.isSecureContext) {
      this.statusText.set('Camera requires a secure context (localhost or HTTPS).');
      this.cameraError.set(
        'Open the app at http://localhost:4200 — not a raw file or insecure host.'
      );
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      this.statusText.set('Camera API unavailable in this browser.');
      this.cameraError.set(
        'Try Chrome or Safari on http://localhost:4200. Embedded previews often block camera access.'
      );
      return;
    }

    try {
      await this.loadModels();

      this.stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });

      const video = this.videoRef?.nativeElement;
      const canvas = this.canvasRef?.nativeElement;
      if (!video || !canvas) {
        this.stream.getTracks().forEach((t) => t.stop());
        this.stream = null;
        this.statusText.set('Camera UI not ready. Refresh and try again.');
        return;
      }

      video.srcObject = this.stream;
      video.muted = true;
      await video.play();

      if (!video.videoWidth || !video.videoHeight) {
        await new Promise<void>((resolve) => {
          video.onloadedmetadata = () => resolve();
        });
      }

      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;

      this.running.set(true);
      this.updateStatus();
      this.loop();
    } catch (err) {
      const name = err instanceof DOMException ? err.name : 'Error';
      const message = err instanceof Error ? err.message : String(err);
      console.error('start failed:', err);

      if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
        this.statusText.set('Camera permission denied.');
        this.cameraError.set(
          'Allow the camera for this site in your browser settings, then click Start camera again.'
        );
      } else if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
        this.statusText.set('No camera found.');
        this.cameraError.set('Connect a camera and try again.');
      } else {
        this.statusText.set('Could not start camera.');
        this.cameraError.set(`${name}: ${message}`);
      }
    }
  }

  stop(): void {
    cancelAnimationFrame(this.rafId);
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.running.set(false);
    this.fps.set(0);
    this.tracks.clear();
    this.frameBuffer = [];
    this.statusText.set('Camera off');
    this.cameraError.set('');
  }

  // ------------------------------------------------------------ main loop

  private loop(): void {
    const video = this.videoRef.nativeElement;
    const canvas = this.canvasRef.nativeElement;
    const ctx = canvas.getContext('2d')!;

    this.zone.runOutsideAngular(() => {
      const tick = () => {
        if (!this.running()) return;

        if (video.readyState >= 2 && this.modelsReady()) {
          const ts = performance.now();
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

          const watched = this.watchedObject();
          const all = this.detector!.detectForVideo(video, ts).detections;
          const relevant = all.filter((d) => {
            const name = d.categories[0].categoryName;
            return name === 'person' || (!!watched && name === watched);
          });

          const hands =
            this.handLandmarker!.detectForVideo(video, ts).landmarks;

          this.updateTracks(relevant);
          this.updateHands(hands, canvas);
          this.checkProximity(ctx);
          this.checkTriggers();

          this.draw(ctx, hands, canvas);
          this.bufferFrame(canvas);
          this.trackFps();
        }

        this.rafId = requestAnimationFrame(tick);
      };
      tick();
    });
  }

  // ------------------------------------------------------------- tracking

  private iou(a: Box, b: Box): number {
    const x1 = Math.max(a.x, b.x);
    const y1 = Math.max(a.y, b.y);
    const x2 = Math.min(a.x + a.w, b.x + b.w);
    const y2 = Math.min(a.y + a.h, b.y + b.h);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    if (inter === 0) return 0;
    return inter / (a.w * a.h + b.w * b.h - inter);
  }

  private updateTracks(detections: any[]): void {
    this.frameNo++;
    const claimed = new Set<number>();

    for (const d of detections) {
      const bb = d.boundingBox!;
      const box: Box = { x: bb.originX, y: bb.originY, w: bb.width, h: bb.height };
      const label = d.categories[0].categoryName;
      const centroid = { x: box.x + box.w / 2, y: box.y + box.h / 2 };

      let bestId = -1;
      let bestScore = CONFIG.iouMatch;

      for (const [id, t] of this.tracks) {
        if (claimed.has(id) || t.label !== label) continue;
        const score = this.iou(box, t.box);
        if (score > bestScore) {
          bestScore = score;
          bestId = id;
        }
      }

      if (bestId !== -1) {
        const t = this.tracks.get(bestId)!;
        t.box = box;
        t.lastSeenFrame = this.frameNo;

        const drift = Math.hypot(
          centroid.x - t.restingCentroid.x,
          centroid.y - t.restingCentroid.y
        );

        if (drift < CONFIG.stillRadius) {
          t.stableFrames++;
        } else if (
          drift > CONFIG.moveRadius &&
          t.stableFrames > CONFIG.stableBeforeMove
        ) {
          this.onObjectMoved(t);
          t.restingCentroid = centroid;
          t.stableFrames = 0;
        }

        claimed.add(bestId);
      } else {
        const id = this.nextTrackId++;
        const displayName =
          label === 'person' ? `Person ${++this.personCount}` : label;

        this.tracks.set(id, {
          id,
          label,
          displayName,
          box,
          lastSeenFrame: this.frameNo,
          handNearFrame: -9999,
          restingCentroid: centroid,
          stableFrames: 0,
        });
        claimed.add(id);
      }
    }

    for (const [id, t] of this.tracks) {
      if (this.frameNo - t.lastSeenFrame > CONFIG.trackExpiry) {
        this.tracks.delete(id);
      }
    }
  }

  // ------------------------------------------------------------ proximity

  private updateHands(hands: any[], canvas: HTMLCanvasElement): void {
    this.handPoints = [];
    for (const hand of hands) {
      // wrist, thumb tip, index tip, middle tip
      for (const i of [0, 4, 8, 12]) {
        this.handPoints.push({
          x: hand[i].x * canvas.width,
          y: hand[i].y * canvas.height,
        });
      }
    }
  }

  private checkProximity(ctx: CanvasRenderingContext2D): void {
    for (const t of this.tracks.values()) {
      if (t.label === 'person') continue;

      const cx = t.box.x + t.box.w / 2;
      const cy = t.box.y + t.box.h / 2;

      let nearest = Infinity;
      for (const p of this.handPoints) {
        const d = Math.hypot(p.x - cx, p.y - cy);
        if (d < nearest) nearest = d;
      }

      if (nearest < CONFIG.handNear) {
        t.handNearFrame = this.frameNo;
      }

      if (CONFIG.showTuning && nearest < Infinity) {
        ctx.fillStyle = nearest < CONFIG.handNear ? '#0b8043' : '#9aa2ad';
        ctx.font = '12px sans-serif';
        ctx.fillText(`${nearest.toFixed(0)}px`, t.box.x, t.box.y + t.box.h + 14);
      }
    }
  }

  /** Which person's box contains this point, if any. */
  private personAt(x: number, y: number): string {
    for (const t of this.tracks.values()) {
      if (t.label !== 'person') continue;
      if (
        x >= t.box.x &&
        x <= t.box.x + t.box.w &&
        y >= t.box.y &&
        y <= t.box.y + t.box.h
      ) {
        return t.displayName;
      }
    }
    return '';
  }

  // -------------------------------------------------------------- trigger

  private checkTriggers(): void {
    if (!this.watchedObject()) return;

    for (const [id, t] of this.tracks) {
      if (t.label === 'person') continue;

      const goneFor = this.frameNo - t.lastSeenFrame;
      const handWasNear = this.frameNo - t.handNearFrame;

      if (
        goneFor > CONFIG.goneMin &&
        goneFor < CONFIG.goneMax &&
        handWasNear < CONFIG.handRecent
      ) {
        const cx = t.box.x + t.box.w / 2;
        const cy = t.box.y + t.box.h / 2;
        this.fireEvent('picked up', t.displayName, this.personAt(cx, cy));
        this.tracks.delete(id);
      }
    }
  }

  private onObjectMoved(t: Track): void {
    const cx = t.box.x + t.box.w / 2;
    const cy = t.box.y + t.box.h / 2;
    this.fireEvent('moved', t.displayName, this.personAt(cx, cy));
  }

  /** Single entry point for every event, automatic or manual. */
  private fireEvent(action: string, object: string, person: string): void {
    const when = new Date();
    const after = this.snapshot();
    const before = this.frameBuffer[0] ?? after;

    if (CONFIG.useFakeCaptions) {
      this.logEvent(action, object, 0.85, when, person, after);
      return;
    }

    void this.captionRemotely(before, after, object, person, when, action);
  }

  private async captionRemotely(
    before: string,
    after: string,
    object: string,
    person: string,
    when: Date,
    fallbackAction: string
  ): Promise<void> {
    try {
      const res = await fetch('/api/caption', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          before: before.split(',')[1],
          after: after.split(',')[1],
          object,
          person,
        }),
      });
      const evt = await res.json();
      this.logEvent(
        evt.action ?? fallbackAction,
        evt.object ?? object,
        evt.confidence ?? 0.5,
        when,
        person,
        after
      );
    } catch (err) {
      console.error('caption failed', err);
      this.logEvent(fallbackAction, object, 0.3, when, person, after);
    }
  }

  // --------------------------------------------------------------- frames

  private snapshot(): string {
    return this.canvasRef.nativeElement.toDataURL('image/jpeg', 0.5);
  }

  private bufferFrame(canvas: HTMLCanvasElement): void {
    if (this.frameNo % CONFIG.bufferEvery !== 0) return;
    this.frameBuffer.push(canvas.toDataURL('image/jpeg', 0.5));
    if (this.frameBuffer.length > CONFIG.bufferSize) {
      this.frameBuffer.shift();
    }
  }

  // -------------------------------------------------------------- drawing

  private draw(
    ctx: CanvasRenderingContext2D,
    hands: any[],
    canvas: HTMLCanvasElement
  ): void {
    for (const t of this.tracks.values()) {
      const isPerson = t.label === 'person';
      ctx.strokeStyle = isPerson ? '#1a73e8' : '#0b8043';
      ctx.lineWidth = 2;
      ctx.strokeRect(t.box.x, t.box.y, t.box.w, t.box.h);

      ctx.fillStyle = ctx.strokeStyle;
      ctx.font = '13px sans-serif';
      ctx.fillText(t.displayName, t.box.x, Math.max(12, t.box.y - 5));
    }

    ctx.fillStyle = '#d93025';
    for (const hand of hands) {
      for (const lm of hand) {
        ctx.beginPath();
        ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  private trackFps(): void {
    const now = performance.now();
    this.frameTimes.push(now);
    while (this.frameTimes.length && now - this.frameTimes[0] > 1000) {
      this.frameTimes.shift();
    }
    const count = this.frameTimes.length;
    if (count !== this.fps()) {
      this.zone.run(() => this.fps.set(count));
    }
  }

  // --------------------------------------------------------------- events

  @HostListener('window:keydown', ['$event'])
  onKey(e: KeyboardEvent): void {
    const tag = (e.target as HTMLElement)?.tagName?.toLowerCase();
    if (tag === 'input' || tag === 'textarea') return;

    if (e.code === 'Space' && this.running()) {
      e.preventDefault();
      this.fireEvent('picked up', this.watchedObject() || 'object', '');
    }
  }

  logEvent(
    action: string,
    object: string,
    confidence: number,
    when: Date = new Date(),
    person = '',
    thumbnail?: string
  ): void {
    const thumb =
      thumbnail ?? (this.running() ? this.snapshot() : '');

    this.zone.run(() => {
      this.events.update((list) => [
        {
          id: this.nextId++,
          time: when,
          action,
          object,
          person,
          confidence,
          thumbnail: thumb,
        },
        ...list,
      ]);
    });
  }

  clearEvents(): void {
    this.events.set([]);
    this.nextId = 1;
  }

  downloadCsv(): void {
    const list = this.events();
    if (!list.length) return;

    const esc = (v: string) => `"${String(v).replace(/"/g, '""')}"`;
    const rows = [
      [
        'id',
        'timestamp_iso',
        'local_time',
        'person',
        'action',
        'object',
        'confidence',
      ],
      ...[...list]
        .sort((a, b) => a.id - b.id)
        .map((e) => [
          String(e.id),
          e.time.toISOString(),
          esc(e.time.toLocaleTimeString()),
          esc(e.person),
          esc(e.action),
          esc(e.object),
          e.confidence.toFixed(2),
        ]),
    ];

    const csv = rows.map((r) => r.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `recallcam-events-${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  ngOnDestroy(): void {
    this.stop();
  }
}