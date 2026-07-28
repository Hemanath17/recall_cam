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
import { IouTracker, Track } from './tracker';

export interface CamEvent {
  id: number;
  time: Date;
  action: string;
  object: string;
  confidence: number;
  thumbnail: string;
}

const CONFIG = {
  /** Low on purpose. Raise it if you get too much junk. */
  scoreThreshold: 0.25,
  /** Cap on simultaneous detections drawn. */
  maxResults: 10,
  /** Must match @mediapipe/tasks-vision in package.json. */
  tasksVisionVersion: '0.10.35',
  /** lite0 is fastest, lite2 is more accurate. Swap and compare. */
  model: 'efficientdet_lite0' as 'efficientdet_lite0' | 'efficientdet_lite2',
  /** Console dump every N frames. Set 0 to silence. */
  logEveryFrames: 60,
};

/** Pixels. A hand within this distance of a box centre counts as "near". */
const HAND_NEAR_PX = 60;
/** Draw the live distance readout while tuning. Turn off for the demo. */
const SHOW_TUNING = true;
/** Frames a track must be missing before a pickup is credible. */
const GONE_MIN = 15;
/** Beyond this, the disappearance is too old to attribute to anything. */
const GONE_MAX = 40;
/** How recently a hand must have been near, in frames. */
const HAND_RECENT = 45;
/** Pixels of displacement that count as a move. */
const MOVE_PX = 40;
/** Frames a track must exist before it is trusted enough to announce. */
const MIN_HITS = 12;
/** Minimum frames between two events for the same track, to avoid spam. */
const EVENT_COOLDOWN = 30;
/** Frames an object must have been still before a move can register. */
const STABLE_BEFORE_MOVE = 20;

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
  readonly statusText = signal('Camera off');
  readonly fps = signal(0);
  readonly events = signal<CamEvent[]>([]);
  readonly query = signal('');
  readonly cameraError = signal('');

  /** Live list of what is on screen right now. */
  readonly seen = signal<string[]>([]);

  /** True when the typed filter matches nothing currently visible. */
  readonly noMatch = signal(false);
  /**
   * Optional filter. Empty shows everything; typing a COCO class name
   * such as "cup", "bottle" or "cell phone" narrows it to that class.
   */
  readonly watchedObject = signal('');

  readonly filteredEvents = computed(() => {
    const q = this.query().trim().toLowerCase();
    const all = this.events();
    if (!q) return all;
    return all.filter((e) =>
      (e.action + ' ' + e.object).toLowerCase().includes(q)
    );
  });

  private stream: MediaStream | null = null;
  private rafId = 0;
  private nextId = 1;
  private frameNo = 0;
  private frameTimes: number[] = [];
  private detector?: ObjectDetector;
  private handLandmarker?: HandLandmarker;
  private modelReady = false;
  private tracker = new IouTracker();

  constructor(private zone: NgZone) {}

  trackById(_: number, e: CamEvent): number {
    return e.id;
  }

  setWatchedObject(value: string): void {
    this.watchedObject.set(value.trim().toLowerCase());
    // Fresh start: re-announce whatever is visible under the new target.
    this.tracker.reset();
  }

  // ---------------------------------------------------------------- model

  private async loadModel(): Promise<void> {
    if (this.modelReady) return;

    this.statusText.set('Loading model…');

    const vision = await FilesetResolver.forVisionTasks(
      `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${CONFIG.tasksVisionVersion}/wasm`
    );

    const modelPath =
      `https://storage.googleapis.com/mediapipe-models/object_detector/` +
      `${CONFIG.model}/float16/1/${CONFIG.model}.tflite`;

    try {
      this.detector = await ObjectDetector.createFromOptions(vision, {
        baseOptions: { modelAssetPath: modelPath, delegate: 'GPU' },
        scoreThreshold: CONFIG.scoreThreshold,
        maxResults: CONFIG.maxResults,
        runningMode: 'VIDEO',
      });
      console.log('detector ready on GPU');
    } catch (gpuError) {
      // Some machines have no working WebGL. CPU is slower but always there.
      console.warn('GPU delegate failed, falling back to CPU', gpuError);
      this.detector = await ObjectDetector.createFromOptions(vision, {
        baseOptions: { modelAssetPath: modelPath, delegate: 'CPU' },
        scoreThreshold: CONFIG.scoreThreshold,
        maxResults: CONFIG.maxResults,
        runningMode: 'VIDEO',
      });
      console.log('detector ready on CPU');
    }

    this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        delegate: 'GPU',
      },
      numHands: 2,
      runningMode: 'VIDEO',
    });
    console.log('hand landmarker ready');

    this.modelReady = true;
  }

  // --------------------------------------------------------------- camera

  async start(): Promise<void> {
    if (this.running()) return;

    this.cameraError.set('');
    this.statusText.set('Requesting camera…');

    if (!window.isSecureContext) {
      this.statusText.set('Camera requires localhost or HTTPS.');
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      this.statusText.set('Camera API unavailable in this browser.');
      return;
    }

    try {
      await this.loadModel();

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
      console.log('canvas', canvas.width, 'x', canvas.height);

      this.running.set(true);
      this.statusText.set('Detecting');
      this.loop();
    } catch (err) {
      const name = err instanceof DOMException ? err.name : 'Error';
      const message = err instanceof Error ? err.message : String(err);
      console.error('start failed:', err);

      if (name === 'NotAllowedError' || name === 'PermissionDeniedError') {
        this.statusText.set('Camera permission denied.');
        this.cameraError.set(
          'Allow the camera for this site, then click Start camera again.'
        );
      } else if (name === 'NotFoundError' || name === 'DevicesNotFoundError') {
        this.statusText.set('No camera found.');
      } else {
        this.statusText.set('Could not start.');
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
    this.seen.set([]);
    this.statusText.set('Camera off');
    this.cameraError.set('');
    this.tracker.reset();
  }

  // ------------------------------------------------------------ main loop

  private loop(): void {
    const video = this.videoRef.nativeElement;
    const canvas = this.canvasRef.nativeElement;
    const ctx = canvas.getContext('2d')!;

    this.zone.runOutsideAngular(() => {
      const tick = () => {
        if (!this.running()) return;

        if (video.readyState >= 2 && this.modelReady) {
          this.frameNo++;
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

          const all =
            this.detector!.detectForVideo(video, performance.now())
              .detections ?? [];

          const hands =
            this.handLandmarker!.detectForVideo(video, performance.now())
              .landmarks ?? [];

          for (const hand of hands) {
            ctx.fillStyle = '#d93025';
            for (const lm of hand) {
              ctx.beginPath();
              ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 4, 0, Math.PI * 2);
              ctx.fill();
            }
          }

          const watched = this.watchedObject();
          const shown = watched
            ? all.filter((d) =>
                (d.categories[0]?.categoryName ?? '').includes(watched)
              )
            : all;

          this.zone.run(() =>
            this.noMatch.set(!!watched && shown.length === 0 && all.length > 0)
          );

          const tracks = this.tracker.update(shown);

          const points = this.handPoints(hands, canvas);
          this.checkProximity(tracks, points, ctx);
          this.checkTriggers(tracks);

          if (this.frameNo % 30 === 0) {
            console.log(
              tracks.map(
                (t) =>
                  `#${t.id} ${t.label} hits=${t.hits} miss=${this.tracker.framesSinceSeen(t)} ann=${t.announced}`
              )
            );
          }

          for (const t of tracks) {
            // Skip tracks that have only been seen once or twice — they are usually
            // detector noise that will vanish next frame.
            if (t.hits < 3) continue;
            this.drawTrack(ctx, t);
          }

          if (
            CONFIG.logEveryFrames &&
            this.frameNo % CONFIG.logEveryFrames === 0
          ) {
            const names = shown.map(
              (d) =>
                `${d.categories[0]?.categoryName ?? '?'} ` +
                `${((d.categories[0]?.score ?? 0) * 100).toFixed(0)}%`
            );
            console.log(`frame ${this.frameNo}:`, names);
            this.zone.run(() => this.seen.set(names));
          }

          this.trackFps();
        }

        this.rafId = requestAnimationFrame(tick);
      };
      tick();
    });
  }

  private drawTrack(ctx: CanvasRenderingContext2D, t: Track): void {
    const isPerson = t.label === 'person';
    const colour = isPerson ? '#1a73e8' : '#0b8043';

    // Fade a track that has not been matched recently.
    const missing = this.tracker.framesSinceSeen(t);
    ctx.globalAlpha = missing > 0 ? 0.4 : 1;

    ctx.strokeStyle = colour;
    ctx.lineWidth = 2;
    ctx.strokeRect(t.box.x, t.box.y, t.box.w, t.box.h);

    const text = `#${t.id} ${t.displayName}`;
    ctx.font = '14px sans-serif';
    const labelWidth = ctx.measureText(text).width + 8;
    const labelY = Math.max(18, t.box.y);

    ctx.fillStyle = colour;
    ctx.fillRect(t.box.x, labelY - 18, labelWidth, 18);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(text, t.box.x + 4, labelY - 5);

    ctx.globalAlpha = 1;
  }

  /**
   * The landmarks worth measuring: wrist, thumb tip, index tip, middle tip.
   * Converted from normalised 0-1 to canvas pixels here, once, so nothing
   * downstream has to think about coordinate spaces.
   */
  private handPoints(
    hands: any[],
    canvas: HTMLCanvasElement
  ): { x: number; y: number }[] {
    const points: { x: number; y: number }[] = [];
    for (const hand of hands) {
      for (const i of [0, 4, 8, 12]) {
        const lm = hand[i];
        if (!lm) continue;
        points.push({ x: lm.x * canvas.width, y: lm.y * canvas.height });
      }
    }
    return points;
  }

  /**
   * Stamp the current frame onto any track a hand is touching. This is the
   * memory that step 4 reads after the object has already gone.
   */
  private checkProximity(
    tracks: Track[],
    points: { x: number; y: number }[],
    ctx: CanvasRenderingContext2D
  ): void {
    for (const t of tracks) {
      if (t.label === 'person') continue;

      let nearest = Infinity;
      for (const p of points) {
        const d = Math.hypot(p.x - t.centroid.x, p.y - t.centroid.y);
        if (d < nearest) nearest = d;
      }

      if (nearest < HAND_NEAR_PX) {
        t.handNearStreak++;
        // Only treat it as contact once it has persisted. A hand passing
        // through the threshold for two frames is not someone touching it.
        if (t.handNearStreak >= 5) {
          t.handNearFrame = this.tracker.currentFrame;
        }
      } else {
        t.handNearStreak = 0;
      }

      if (SHOW_TUNING && nearest < Infinity) {
        ctx.fillStyle = nearest < HAND_NEAR_PX ? '#0b8043' : '#9aa2ad';
        ctx.font = '12px sans-serif';
        ctx.fillText(
          `${nearest.toFixed(0)}px`,
          t.box.x,
          t.box.y + t.box.h + 14
        );
      }
    }
  }

  /**
   * A track of the same class that has gone unmatched for a while. When an
   * object is carried across the frame it disappears and reappears faster
   * than the old track expires, so the stale track is the evidence that this
   * is the same object rather than a new one.
   */
  private findStalePredecessor(
    label: string,
    exceptId: number
  ): Track | undefined {
    for (const t of this.tracker.all()) {
      if (t.id === exceptId) continue;
      if (t.label !== label) continue;
      if (!t.announced) continue;
      if (this.tracker.framesSinceSeen(t) < 5) continue;
      return t;
    }
    return undefined;
  }

  /**
   * Logs one "detected" per track when it first establishes, then a "moved"
   * every time it is displaced from where it settled. Runs only while an
   * object name is typed; changing that name resets everything.
   */
  private checkTriggers(tracks: Track[]): void {
    const watched = this.watchedObject();
    if (!watched) return;

    const now = this.tracker.currentFrame;

    for (const t of tracks) {
      if (t.label === 'person') continue;
      // Only the class the user asked for.
      if (!t.label.includes(watched)) continue;
      // Ignore brand-new tracks; most are detector noise that vanishes.
      if (t.hits < MIN_HITS) continue;
      // Only act on tracks matched this frame.
      if (this.tracker.framesSinceSeen(t) !== 0) continue;

      // First sighting.
      if (!t.announced) {
        t.announced = true;
        t.lastEventFrame = now;

        const stale = this.findStalePredecessor(t.label, t.id);
        const ghost = stale
          ? { centroid: stale.centroid }
          : this.tracker.findGhost(t.label);

        if (ghost) {
          const distance = Math.hypot(
            t.centroid.x - ghost.centroid.x,
            t.centroid.y - ghost.centroid.y
          );
          if (stale) this.tracker.remove(stale.id);
          this.tracker.clearGhosts(t.label);

          this.logEvent(
            distance > MOVE_PX ? 'moved' : 'returned',
            t.displayName,
            0.75
          );
        } else {
          this.logEvent('detected', t.displayName, 0.9);
        }
        continue;
      }

      if (now - t.lastEventFrame < EVENT_COOLDOWN) continue;

      if (t.stableFrames === 0 && t.lastDrift > MOVE_PX) {
        t.lastEventFrame = now;
        t.lastDrift = 0;
        this.logEvent('moved', t.displayName, 0.75);
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
      this.logEvent(
        'detected',
        this.watchedObject() || this.seen().join(', ') || 'nothing',
        1
      );
    }
  }

  logEvent(
    action: string,
    object: string,
    confidence: number,
    when: Date = new Date()
  ): void {
    const canvas = this.canvasRef.nativeElement;
    const thumb = this.running() ? canvas.toDataURL('image/jpeg', 0.5) : '';
    this.zone.run(() => {
      this.events.update((list) => [
        {
          id: this.nextId++,
          time: when,
          action,
          object,
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
      ['id', 'timestamp_iso', 'local_time', 'action', 'object', 'confidence'],
      ...[...list]
        .sort((a, b) => a.id - b.id)
        .map((e) => [
          String(e.id),
          e.time.toISOString(),
          esc(e.time.toLocaleTimeString()),
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