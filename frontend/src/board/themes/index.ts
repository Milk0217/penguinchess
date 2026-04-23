import type { BoardTheme } from "../types";
import { defaultTheme } from "./default";
import { darkTheme } from "./dark";

/** 主题注册表 */
export const THEMES: Record<string, BoardTheme> = {
  default: defaultTheme,
  dark: darkTheme,
};

/** 根据 id 获取主题，未找到时回退到默认主题 */
export function getTheme(id: string): BoardTheme {
  return THEMES[id] ?? defaultTheme;
}

/** 获取所有已注册主题列表 */
export function getAllThemes(): BoardTheme[] {
  return Object.values(THEMES);
}

// Re-export individual themes for convenience
export { defaultTheme } from "./default";
export { darkTheme } from "./dark";
