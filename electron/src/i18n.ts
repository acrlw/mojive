import { useSyncExternalStore } from 'react';
import chinese from './locales/zh.json';
export type Language = 'zh' | 'en';
let language: Language = 'zh';
const listeners = new Set<() => void>();
const catalogue: Record<string, string> = chinese;
export function setLanguage(value: Language) {
  if (language === value) return;
  language = value;
  for (const listener of listeners) listener();
}
export function getLanguage() {
  return language;
}
export function useLanguage() {
  return useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    getLanguage,
    () => 'zh' as Language,
  );
}
const prefixes: [string, string][] = [
  ['Updated keyframe ', '已更新关键帧 '],
  ['Added keyframe ', '已添加关键帧 '],
  ['Loaded ', '已加载 '],
  ['Opened ', '已打开 '],
  ['Saved ', '已保存 '],
  ['Added ', '已添加 '],
  ['Removed ', '已删除 '],
  ['Deleted ', '已删除 '],
  ['Hid ', '已隐藏 '],
  ['Showed ', '已显示 '],
  ['Imported ', '已导入 '],
  ['Failed to load ', '加载失败：'],
  ['Could not open ', '无法打开 '],
];
export function t(text: string, locale: Language = language): string {
  if (locale === 'en' || !text) return text;
  if (catalogue[text]) return catalogue[text];
  const axisLabel = text.match(/^(.+) ([XYZ])$/);
  if (axisLabel && catalogue[axisLabel[1]]) return catalogue[axisLabel[1]] + ' ' + axisLabel[2];
  const loaded = text.match(/^Loaded (.+) · (\d+) objects · (\d+) joints$/);
  if (loaded) return `已加载 ${loaded[1]} · ${loaded[2]} 个对象 · ${loaded[3]} 个关节`;
  const folder = text.match(/^Opened project folder (.+)$/);
  if (folder) return '已打开项目文件夹 ' + folder[1];
  const added = text.match(/^Added (box|sphere|cylinder|capsule|ground|light|camera)$/);
  if (added)
    return (
      '已添加' +
      t(
        added[1] === 'light' ? 'Light source' : added[1][0].toUpperCase() + added[1].slice(1),
        locale,
      )
    );
  for (const [prefix, translated] of prefixes)
    if (text.startsWith(prefix)) return translated + text.slice(prefix.length);
  const count = text.match(/^(\d+) (objects?|joints?|files?|keyframes?)$/);
  if (count)
    return (
      count[1] +
      ' ' +
      (
        {
          object: '个对象',
          objects: '个对象',
          joint: '个关节',
          joints: '个关节',
          file: '个文件',
          files: '个文件',
          keyframe: '个关键帧',
          keyframes: '个关键帧',
        } as Record<string, string>
      )[count[2]]
    );
  return text;
}
