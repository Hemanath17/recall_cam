/**
 * Rolling-window statistics about detector behaviour, per object class.
 *
 * The point of this is to answer questions the trigger logic depends on:
 * does the cup stay detected while it sits still? Does it drop out the moment
 * a hand covers it? Is one physical object producing one track or six?
 *
 * Call record() once per frame with the UNFILTERED detections, then read
 * report() whenever you want to look.
 */

export interface ClassStats {
    className: string;
    /** Fraction of the window in which this class was detected, 0 to 1. */
    presenceRate: number;
    meanScore: number;
    minScore: number;
    /** Every absence that has since ended, in frames. */
    closedGaps: number[];
    /** Frames since last seen. 0 when present this frame. */
    openGap: number;
    /** Distinct track IDs first seen inside the window. */
    trackChurn: number;
    /** Convenience percentiles over closedGaps. */
    medianGap: number;
    p95Gap: number;
    maxClosedGap: number;
  }
  
  interface Sample {
    frame: number;
    score: number;
  }
  
  interface ClassRecord {
    samples: Sample[];
    lastSeen: number;
    openGap: number;
    closedGaps: number[];
    /** trackId -> frame it was first observed. */
    trackFirstSeen: Map<number, number>;
  }
  
  export class DiagnosticsRecorder {
    private classes = new Map<string, ClassRecord>();
    private currentFrame = 0;
  
    constructor(private windowFrames = 150) {}
  
    /**
     * @param frameNo    monotonically increasing frame counter
     * @param detections raw MediaPipe detections, before any class filtering
     * @param tracks     your live track map, used for churn only
     */
    record(frameNo: number, detections: any[], tracks: Map<number, any>): void {
      this.currentFrame = frameNo;
      const cutoff = frameNo - this.windowFrames;
  
      // --- 1. what appeared this frame, and at what confidence ---------------
      const seenNow = new Map<string, number>();
      for (const d of detections) {
        const category = d.categories?.[0];
        if (!category) continue;
        const name = category.categoryName as string;
        const score = category.score as number;
        // Keep the strongest instance if a class appears more than once.
        const existing = seenNow.get(name);
        if (existing === undefined || score > existing) {
          seenNow.set(name, score);
        }
      }
  
      // --- 2. classes present this frame: close any open gap -----------------
      for (const [name, score] of seenNow) {
        const rec = this.ensure(name);
        if (rec.openGap > 0) {
          rec.closedGaps.push(rec.openGap);
          rec.openGap = 0;
        }
        rec.lastSeen = frameNo;
        rec.samples.push({ frame: frameNo, score });
      }
  
      // --- 3. classes absent this frame: widen the open gap ------------------
      // Easy to forget. Without this, gaps never grow.
      for (const [name, rec] of this.classes) {
        if (!seenNow.has(name)) rec.openGap++;
      }
  
      // --- 4. track churn ----------------------------------------------------
      for (const [id, t] of tracks) {
        const name: string = t.label;
        const rec = this.ensure(name);
        if (!rec.trackFirstSeen.has(id)) {
          rec.trackFirstSeen.set(id, frameNo);
        }
      }
  
      // --- 5. drop anything that has aged out of the window -------------------
      for (const rec of this.classes.values()) {
        while (rec.samples.length && rec.samples[0].frame < cutoff) {
          rec.samples.shift();
        }
        for (const [id, firstFrame] of rec.trackFirstSeen) {
          if (firstFrame < cutoff) rec.trackFirstSeen.delete(id);
        }
      }
    }
  
    report(): ClassStats[] {
      const out: ClassStats[] = [];
  
      for (const [className, rec] of this.classes) {
        const scores = rec.samples.map((s) => s.score);
        const sortedGaps = [...rec.closedGaps].sort((a, b) => a - b);
  
        out.push({
          className,
          presenceRate: Math.min(1, rec.samples.length / this.windowFrames),
          meanScore: scores.length
            ? scores.reduce((a, b) => a + b, 0) / scores.length
            : 0,
          minScore: scores.length ? Math.min(...scores) : 0,
          closedGaps: rec.closedGaps,
          openGap: rec.openGap,
          trackChurn: rec.trackFirstSeen.size,
          medianGap: percentile(sortedGaps, 0.5),
          p95Gap: percentile(sortedGaps, 0.95),
          maxClosedGap: sortedGaps.length ? sortedGaps[sortedGaps.length - 1] : 0,
        });
      }
  
      // Most-present classes first, so the interesting rows are at the top.
      return out.sort((a, b) => b.presenceRate - a.presenceRate);
    }
  
    /** One-line-per-class text summary, handy for console or canvas. */
    reportLines(): string[] {
      return this.report().map(
        (s) =>
          `${s.className.padEnd(12)} ` +
          `seen ${(s.presenceRate * 100).toFixed(0).padStart(3)}%  ` +
          `score ${s.meanScore.toFixed(2)}/${s.minScore.toFixed(2)}  ` +
          `gap med ${s.medianGap} p95 ${s.p95Gap} max ${s.maxClosedGap}  ` +
          `open ${s.openGap}  tracks ${s.trackChurn}`
      );
    }
  
    reset(): void {
      this.classes.clear();
      this.currentFrame = 0;
    }
  
    private ensure(name: string): ClassRecord {
      let rec = this.classes.get(name);
      if (!rec) {
        rec = {
          samples: [],
          lastSeen: -1,
          openGap: 0,
          closedGaps: [],
          trackFirstSeen: new Map(),
        };
        this.classes.set(name, rec);
      }
      return rec;
    }
  }
  
  function percentile(sorted: number[], p: number): number {
    if (!sorted.length) return 0;
    const idx = Math.min(sorted.length - 1, Math.floor(p * sorted.length));
    return sorted[idx];
  }