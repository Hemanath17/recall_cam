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
import { ObjectDetector, FilesetResolver } from '@mediapipe/tasks-vision';

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
  private modelReady = false;

  constructor(private zone: NgZone) {}

  trackById(_: number, e: CamEvent): number {
    return e.id;
  }

  setWatchedObject(value: string): void {
    this.watchedObject.set(value.trim().toLowerCase());
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

          const watched = this.watchedObject();
          const shown = watched
            ? all.filter((d) =>
                (d.categories[0]?.categoryName ?? '').includes(watched)
              )
            : all;

          this.zone.run(() =>
            this.noMatch.set(!!watched && shown.length === 0 && all.length > 0)
          );

          for (const d of shown) {
            this.drawDetection(ctx, d);
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

  private drawDetection(ctx: CanvasRenderingContext2D, d: any): void {
    const b = d.boundingBox;
    if (!b) return;

    const category = d.categories?.[0];
    const name = category?.categoryName || '(unlabelled)';
    const score = category?.score ?? 0;
    const isPerson = name === 'person';
    const colour = isPerson ? '#1a73e8' : '#0b8043';

    ctx.strokeStyle = colour;
    ctx.lineWidth = 2;
    ctx.strokeRect(b.originX, b.originY, b.width, b.height);

    const text = `${name} ${(score * 100).toFixed(0)}%`;
    ctx.font = '14px sans-serif';
    const labelWidth = ctx.measureText(text).width + 8;
    const labelY = Math.max(18, b.originY);

    ctx.fillStyle = colour;
    ctx.fillRect(b.originX, labelY - 18, labelWidth, 18);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(text, b.originX + 4, labelY - 5);
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