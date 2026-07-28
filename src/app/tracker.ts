/**
 * IoU tracker.
 *
 * An object detector is stateless: it re-recognises the scene from scratch on
 * every frame and hands back an unlabelled list of boxes. That is enough to
 * draw rectangles, but not to say anything about time. "The phone was here and
 * now it is gone" needs an identity that survives between frames, and this is
 * the layer that maintains one.
 *
 * The assumption is simply that things move slowly relative to the frame rate.
 * At 30fps an object shifts a few pixels per frame, so a new box that heavily
 * overlaps a box from last frame is almost certainly the same object.
 */

export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Track {
  /** Stable across frames. This is the whole point of the class. */
  id: number;
  /** COCO class name, e.g. "cell phone". */
  label: string;
  /** Shown in the UI. Persons get numbered, objects keep their class name. */
  displayName: string;
  box: Box;
  /** Centre of the current box, recomputed each update. */
  centroid: { x: number; y: number };
  /** Frame this track was created. */
  firstFrame: number;
  /** Most recent frame this track was matched to a detection. */
  lastSeenFrame: number;
  /** Consecutive frames matched. Low values mean a shaky, new track. */
  hits: number;
  /**
   * Where the object settled. Displacement is measured from here rather than
   * from the previous frame, so slow drift does not accumulate into a false
   * "moved" and jitter does not register at all.
   */
  restingCentroid: { x: number; y: number };
  /** Frames spent within stillRadius of restingCentroid. */
  stableFrames: number;
  /**
   * Last frame a hand was within the proximity threshold. Stays set after the
   * hand leaves, which is the point: when a track ends, the hand is already
   * gone and this is the only surviving evidence.
   */
  handNearFrame: number;
  /** Consecutive frames a hand has been within the threshold. */
  handNearStreak: number;
  /** True once a "detected" event has been logged for this track. */
  announced: boolean;
  /** Frame of the last event logged for this track. Used as a cooldown. */
  lastEventFrame: number;
  /** How far the object jumped on the frame it left its resting place. */
  lastDrift: number;
}

export interface TrackerConfig {
  /** Minimum overlap to consider two boxes the same object. */
  iouMatch: number;
  /** Frames a track survives unmatched before being deleted. */
  maxAge: number;
  /** Pixel drift from restingCentroid still counted as stationary. */
  stillRadius: number;
}

export const DEFAULT_TRACKER_CONFIG: TrackerConfig = {
  iouMatch: 0.15,   // was 0.3 — tolerate fast movement
  maxAge: 150,      // was 60 — survive five seconds of dropout
  stillRadius: 15,
};

/**
 * Intersection over Union.
 *
 * The overlapping area of two boxes divided by the total area they jointly
 * cover. 1.0 means identical, 0 means no overlap at all. Dividing by the union
 * rather than by either box's area is what makes it scale-independent: a small
 * box sitting inside a large one does not score highly, which matters because
 * detectors often emit nested boxes for the same thing.
 */
export function iou(a: Box, b: Box): number {
  const left = Math.max(a.x, b.x);
  const top = Math.max(a.y, b.y);
  const right = Math.min(a.x + a.w, b.x + b.w);
  const bottom = Math.min(a.y + a.h, b.y + b.h);

  const overlapWidth = right - left;
  const overlapHeight = bottom - top;
  if (overlapWidth <= 0 || overlapHeight <= 0) return 0;

  const intersection = overlapWidth * overlapHeight;
  const union = a.w * a.h + b.w * b.h - intersection;
  return union <= 0 ? 0 : intersection / union;
}

interface Ghost {
  label: string;
  centroid: { x: number; y: number };
  diedFrame: number;
  id: number;
}

export class IouTracker {
  private tracks = new Map<number, Track>();
  private nextId = 1;
  private personCount = 0;
  private frameNo = 0;
  private ghosts: Ghost[] = [];
  /** How long a dead track is remembered, in frames. */
  private ghostTtl = 300;

  constructor(private config: TrackerConfig = DEFAULT_TRACKER_CONFIG) {}

  get currentFrame(): number {
    return this.frameNo;
  }

  /** Live tracks, in creation order. */
  all(): Track[] {
    return [...this.tracks.values()];
  }

  get(id: number): Track | undefined {
    return this.tracks.get(id);
  }

  remove(id: number): void {
    this.tracks.delete(id);
  }

  reset(): void {
    this.tracks.clear();
    this.ghosts = [];
    this.nextId = 1;
    this.personCount = 0;
    this.frameNo = 0;
  }

  /**
   * Feed one frame of detections. Call exactly once per frame, then read
   * all() to draw or to run trigger rules.
   *
   * @param detections raw MediaPipe detections for this frame
   */
  update(detections: any[]): Track[] {
    this.frameNo++;

    // A track can only absorb one detection per frame, otherwise two nearby
    // boxes of the same class would both claim the same identity.
    const claimed = new Set<number>();

    for (const detection of detections) {
      const bb = detection.boundingBox;
      const category = detection.categories?.[0];
      if (!bb || !category) continue;

      const box: Box = {
        x: bb.originX,
        y: bb.originY,
        w: bb.width,
        h: bb.height,
      };
      const label: string = category.categoryName ?? 'unknown';
      const centroid = { x: box.x + box.w / 2, y: box.y + box.h / 2 };

      const matchId = this.findBestMatch(box, label, claimed);

      if (matchId !== null) {
        this.absorb(matchId, box, centroid);
        claimed.add(matchId);
      } else {
        const id = this.create(label, box, centroid);
        claimed.add(id);
      }
    }

    this.expireStale();
    return this.all();
  }

  /**
   * The best existing track for this detection, or null if none is close
   * enough. Only tracks of the same class are considered: a phone should never
   * inherit a cup's identity no matter how well the boxes line up.
   */
  private findBestMatch(
    box: Box,
    label: string,
    claimed: Set<number>
  ): number | null {
    let bestId: number | null = null;
    let bestScore = this.config.iouMatch;

    for (const [id, track] of this.tracks) {
      if (claimed.has(id)) continue;
      if (track.label !== label) continue;

      const score = iou(box, track.box);
      if (score > bestScore) {
        bestScore = score;
        bestId = id;
      }
    }

    return bestId;
  }

  /** An existing track takes on this frame's box. */
  private absorb(
    id: number,
    box: Box,
    centroid: { x: number; y: number }
  ): void {
    const track = this.tracks.get(id)!;

    track.box = box;
    track.centroid = centroid;
    track.lastSeenFrame = this.frameNo;
    track.hits++;

    const drift = Math.hypot(
      centroid.x - track.restingCentroid.x,
      centroid.y - track.restingCentroid.y
    );

    if (drift < this.config.stillRadius) {
      track.stableFrames++;
    } else {
      // Record the displacement before overwriting the resting point,
      // otherwise it is lost and nothing downstream can see the move.
      track.lastDrift = drift;
      track.restingCentroid = { ...centroid };
      track.stableFrames = 0;
    }
  }

  private create(
    label: string,
    box: Box,
    centroid: { x: number; y: number }
  ): number {
    const id = this.nextId++;
    const displayName =
      label === 'person' ? `Person ${++this.personCount}` : label;

    this.tracks.set(id, {
      id,
      label,
      displayName,
      box,
      centroid,
      firstFrame: this.frameNo,
      lastSeenFrame: this.frameNo,
      hits: 1,
      restingCentroid: { ...centroid },
      stableFrames: 0,
      handNearFrame: -9999,
      handNearStreak: 0,
      announced: false,
      lastEventFrame: -9999,
      lastDrift: 0,
    });

    return id;
  }

  /**
   * Tracks are kept alive for maxAge frames after they were last matched.
   *
   * This tolerance is deliberate. Detectors flicker: an object can vanish for
   * two or three frames and reappear unchanged. Deleting on the first miss
   * would mint a fresh ID every time that happened, and every conclusion about
   * "this object disappeared" would be noise.
   */
  private expireStale(): void {
    for (const [id, track] of this.tracks) {
      if (this.frameNo - track.lastSeenFrame > this.config.maxAge) {
        this.ghosts.push({
          label: track.label,
          centroid: { ...track.centroid },
          diedFrame: this.frameNo,
          id: track.id,
        });
        this.tracks.delete(id);
      }
    }
    this.ghosts = this.ghosts.filter(
      (g) => this.frameNo - g.diedFrame < this.ghostTtl
    );
  }

  /**
   * A recently dead track of the same class. Used to tell "the object was
   * occluded and came back" from "a new object appeared".
   */
  findGhost(label: string): Ghost | undefined {
    for (let i = this.ghosts.length - 1; i >= 0; i--) {
      if (this.ghosts[i].label === label) return this.ghosts[i];
    }
    return undefined;
  }

  clearGhosts(label: string): void {
    this.ghosts = this.ghosts.filter((g) => g.label !== label);
  }

  /** Frames since a track was last matched. 0 means seen this frame. */
  framesSinceSeen(track: Track): number {
    return this.frameNo - track.lastSeenFrame;
  }
}
