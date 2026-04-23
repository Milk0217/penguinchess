// ============================================================
// PenguinChess — Board Layout Registry
// ============================================================

import type { BoardLayout } from "../types";
import { parallelogramLayout } from "./parallelogram";
import { hexagonLayout } from "./hexagon";

/** All registered layouts keyed by layout ID. */
const layoutMap = new Map<string, BoardLayout>();

function register(layout: BoardLayout): void {
  layoutMap.set(layout.id, layout);
}

// Register built-in layouts
register(parallelogramLayout);
register(hexagonLayout);

/** Look up a layout by its ID. Returns undefined if not found. */
export function getLayout(id: string): BoardLayout | undefined {
  return layoutMap.get(id);
}

/** Return all registered layouts as an array. */
export function getAllLayouts(): BoardLayout[] {
  return Array.from(layoutMap.values());
}

/** The default layout used when none is specified. */
export const defaultLayout: BoardLayout = parallelogramLayout;

export { parallelogramLayout, hexagonLayout };
