import type { Scene, SceneReport, Timestamp, TimelineScene } from "../types";

export function cn(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(" ");
}

export function timestampSeconds(timestamp?: Timestamp | null): number | null {
  if (!timestamp) {
    return null;
  }
  if (typeof timestamp.seconds === "number") {
    return timestamp.seconds;
  }
  if (!timestamp.timebase || timestamp.timebase.den === 0) {
    return null;
  }
  return timestamp.pts * (timestamp.timebase.num / timestamp.timebase.den);
}

export function sceneStartSeconds(scene: TimelineScene): number {
  if (isSceneReport(scene)) {
    return scene.start_seconds;
  }
  return timestampSeconds(scene.start.timestamp) ?? 0;
}

export function sceneEndSeconds(scene: TimelineScene): number {
  if (isSceneReport(scene)) {
    return scene.end_seconds;
  }
  return timestampSeconds(scene.end.timestamp) ?? sceneStartSeconds(scene);
}

export function sceneStartFrame(scene: TimelineScene): number {
  return isSceneReport(scene) ? scene.start_frame : scene.start.frame_index;
}

export function sceneEndFrame(scene: TimelineScene): number {
  return isSceneReport(scene) ? scene.end_frame : scene.end.frame_index;
}

export function sceneIndex(scene: TimelineScene, fallback: number): number {
  return isSceneReport(scene) ? scene.index : fallback;
}

export function formatSeconds(value?: number | null): string {
  if (value == null || !Number.isFinite(value)) {
    return "n/a";
  }
  const totalMs = Math.max(0, Math.round(value * 1000));
  const totalSeconds = Math.floor(totalMs / 1000);
  const ms = totalMs % 1000;
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  return `${hours.toString().padStart(2, "0")}:${minutes
    .toString()
    .padStart(2, "0")}:${seconds.toString().padStart(2, "0")}.${ms
    .toString()
    .padStart(3, "0")}`;
}

export function formatNumber(value?: number | null): string {
  if (value == null || !Number.isFinite(value)) {
    return "n/a";
  }
  return new Intl.NumberFormat("en-US").format(value);
}

export function formatBytes(value?: number | null): string {
  if (value == null || !Number.isFinite(value)) {
    return "n/a";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  let next = Math.max(0, value);
  let unit = 0;
  while (next >= 1024 && unit < units.length - 1) {
    next /= 1024;
    unit += 1;
  }
  return `${next.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

export function formatScore(value?: number | null): string {
  if (value == null || !Number.isFinite(value)) {
    return "n/a";
  }
  if (value >= 0 && value <= 1) {
    return `${Math.round(value * 100)}%`;
  }
  return value.toFixed(2);
}

export function ratioPercent(value: number, max: number): number {
  if (!Number.isFinite(value) || !Number.isFinite(max) || max <= 0) {
    return 0;
  }
  return Math.max(0, Math.min(100, (value / max) * 100));
}

export function isSceneReport(scene: Scene | SceneReport): scene is SceneReport {
  return "start_seconds" in scene;
}
