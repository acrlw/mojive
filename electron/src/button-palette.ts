/** Source: python/ui/interaction_paint.py primary_action_ink/secondary_action_ink. */
type RGB = readonly [number, number, number];
const rgb = (hex: string): RGB =>
  [1, 3, 5].map((index) => parseInt(hex.slice(index, index + 2), 16)) as unknown as RGB;
const luminance = (color: RGB) =>
  color
    .map((channel) => {
      const value = channel / 255;
      return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
    })
    .reduce((sum, value, index) => sum + value * [0.2126, 0.7152, 0.0722][index], 0);
const roundEven = (value: number) => {
  const floor = Math.floor(value);
  return value - floor === 0.5 ? (floor % 2 === 0 ? floor : floor + 1) : Math.round(value);
};
const mix = (first: string, second: string, amount: number) =>
  '#' +
  rgb(first)
    .map((channel, index) =>
      roundEven(channel + (rgb(second)[index] - channel) * amount)
        .toString(16)
        .padStart(2, '0'),
    )
    .join('')
    .toUpperCase();
export function buttonPalette(palette: Record<string, string>) {
  const light = luminance(rgb(palette.on_primary)) > luminance(rgb(palette.primary));
  const secondary = (field: string) =>
    luminance(rgb(field)) > luminance(rgb(palette.text)) ? mix(field, palette.text, 0.12) : field;
  return {
    'button-primary': palette.primary,
    'button-primary-hover': light
      ? mix(palette.primary, palette.primary_bright, 0.5)
      : palette.primary_dim,
    'button-primary-pressed': palette.primary_bright,
    'button-primary-text': palette.on_primary,
    'button-secondary': secondary(palette.field),
    'button-secondary-hover': secondary(palette.field_hovered),
    'button-secondary-pressed': secondary(palette.field_active),
  };
}
