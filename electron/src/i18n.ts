import { createInstance } from 'i18next';
import { initReactI18next, useTranslation } from 'react-i18next';
import nativeChinese from './locales/zh.json';
import electronChinese from './locales/electron-zh.json';

const chinese = { ...nativeChinese, ...electronChinese };

export type Language = 'zh' | 'en';
export type MessageValues = Record<string, string | number | LocalizedMessage>;
export interface LocalizedMessage {
  key: string;
  values?: MessageValues;
}
export type LocalizedText = string | LocalizedMessage;

export const i18n = createInstance();
void i18n.use(initReactI18next).init({
  initAsync: false,
  lng: 'zh',
  fallbackLng: 'en',
  supportedLngs: ['zh', 'en'],
  keySeparator: false,
  nsSeparator: false,
  resources: {
    zh: { translation: chinese },
    en: { translation: Object.fromEntries(Object.keys(chinese).map((key) => [key, key])) },
  },
  interpolation: { escapeValue: false, skipOnVariables: true },
  react: { useSuspense: false },
});

export function setLanguage(value: Language) {
  if (getLanguage() !== value) void i18n.changeLanguage(value);
  document.documentElement.lang = value === 'zh' ? 'zh-CN' : 'en';
}
export function getLanguage(): Language {
  return i18n.language === 'en' ? 'en' : 'zh';
}
export function useLanguage(): Language {
  useTranslation(undefined, { i18n });
  return getLanguage();
}
/** Store the key and arguments so existing output and notifications switch languages too. */
export function message(key: string, values?: MessageValues): LocalizedMessage {
  return { key, values };
}
export function t(text: LocalizedText, locale?: Language): string;
export function t(text: string, values?: MessageValues): string;
export function t(text: LocalizedText, options?: Language | MessageValues): string {
  const locale = typeof options === 'string' ? options : getLanguage();
  const key = typeof text === 'string' ? text : text.key;
  if (!key) return '';
  const values =
    typeof text === 'string' ? (typeof options === 'object' ? options : {}) : text.values;
  const resolved = Object.fromEntries(
    Object.entries(values ?? {}).map(([name, value]) => [
      name,
      typeof value === 'object' ? t(value, locale) : value,
    ]),
  );
  return String(i18n.t(key, { ...resolved, lng: locale, defaultValue: key }));
}
