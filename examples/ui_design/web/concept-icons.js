// Original cc-design glyphs; 24-unit grid, 1.75-unit stroke.
export const conceptIcons = {
  move: '<path d="M12 3v18M3 12h18M12 3l-3 3M12 3l3 3M12 21l-3-3M12 21l3-3M3 12l3-3M3 12l3 3M21 12l-3-3M21 12l-3 3"/>',
  rotate:
    '<circle cx="12" cy="12" r="8"/><ellipse cx="12" cy="12" rx="3" ry="8"/><ellipse cx="12" cy="12" rx="8" ry="3"/>',
  scale:
    '<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M9 15l6-6M11 9h4v4"/>',
  perturb:
    '<circle cx="7" cy="17" r="2.2"/><circle cx="17" cy="7" r="2.2"/><path d="M8.6 15.4l6.8-6.8" stroke-dasharray="2 2"/>',
  cube: '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M4 7.5l8 4.5 8-4.5M12 12v9"/>',
  magnet: '<path d="M6 4v8a6 6 0 0 0 12 0V4"/><path d="M6 8h3M15 8h3"/>',
  play: '<path d="M8 5.5v13l10.5-6.5z" fill="currentColor" stroke="none"/>',
  pause:
    '<rect x="7" y="5.5" width="3.2" height="13" rx="1" fill="currentColor" stroke="none"/><rect x="13.8" y="5.5" width="3.2" height="13" rx="1" fill="currentColor" stroke="none"/>',
  stepb: '<path d="M15 6l-6 6 6 6"/>',
  stepf: '<path d="M9 6l6 6-6 6"/>',
  reset: '<path d="M4 12a8 8 0 1 0 2.4-5.7"/><path d="M4 4v4h4"/>',
  rec: '<circle cx="12" cy="12" r="6" fill="currentColor" stroke="none"/>',
  stop: '<rect x="7" y="7" width="10" height="10" rx="2" fill="currentColor" stroke="none"/>',
  chev: '<path d="M6 9l6 6 6-6"/>',
  search: '<circle cx="11" cy="11" r="6.5"/><path d="M20 20l-4.2-4.2"/>',
  filter: '<path d="M4 6h16M7 12h10M10 18h4"/>',
  eye: '<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z"/><circle cx="12" cy="12" r="2.8"/>',
  eyeoff:
    '<path d="M3 3l18 18M10.6 5.6A9.9 9.9 0 0 1 12 5.5c6 0 9.5 6.5 9.5 6.5a17 17 0 0 1-3.1 3.9M6.1 6.9C3.8 8.6 2.5 12 2.5 12S6 18.5 12 18.5c1.6 0 3-.4 4.2-1"/>',
  lock: '<rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/>',
  more: '<circle cx="5.5" cy="12" r="1.3" fill="currentColor"/><circle cx="12" cy="12" r="1.3" fill="currentColor"/><circle cx="18.5" cy="12" r="1.3" fill="currentColor"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  x: '<path d="M6 6l12 12M18 6L6 18"/>',
  tree: '<path d="M5 4v16M5 8h6M5 14h6"/><rect x="11" y="5.5" width="8" height="5" rx="1.2"/><rect x="11" y="11.5" width="8" height="5" rx="1.2"/>',
  box: '<path d="M21 8l-9-5-9 5 9 5 9-5zM3 8v8l9 5 9-5V8M12 13v8"/>',
  joint:
    '<circle cx="12" cy="12" r="3"/><path d="M12 3v6M12 15v6M4.2 7.5l5.2 3M14.6 13.5l5.2 3"/>',
  sliders:
    '<path d="M4 7h10M18 7h2M4 17h4M12 17h8"/><circle cx="16" cy="7" r="2"/><circle cx="10" cy="17" r="2"/>',
  camera:
    '<path d="M4 8h3l2-2.5h6L17 8h3v11H4z"/><circle cx="12" cy="13" r="3.5"/>',
  folder:
    '<path d="M3 6.5A1.5 1.5 0 0 1 4.5 5H10l2 2.5h7.5A1.5 1.5 0 0 1 21 9v9.5a1.5 1.5 0 0 1-1.5 1.5h-15A1.5 1.5 0 0 1 3 18.5z"/>',
  gear: '<circle cx="12" cy="12" r="2.4545"/><path d="M18.0545 14.4545C17.8436 14.9515 17.9393 15.5256 18.3 15.9273L18.3818 16.0091C18.894 16.3933 19.1352 17.0409 18.9989 17.6665C18.8626 18.2921 18.3739 18.7808 17.7483 18.917C17.1227 19.0533 16.4751 18.8122 16.0909 18.3L16.0091 18.2182C15.6074 17.8575 15.0333 17.7618 14.5364 17.9727C14.0513 18.1916 13.7337 18.6681 13.7182 19.2V19.3636C13.7182 20.2674 12.9856 21 12.0818 21C11.1781 21 10.4455 20.2674 10.4455 19.3636V19.2818C10.4157 18.7303 10.0625 18.2487 9.5455 18.0545C9.0485 17.8436 8.4744 17.9393 8.0727 18.3L7.9909 18.3818C7.6067 18.894 6.9591 19.1352 6.3335 18.9989C5.7079 18.8626 5.2192 18.3739 5.083 17.7483C4.9467 17.1227 5.1878 16.4751 5.7 16.0909L5.7818 16.0091C6.1425 15.6074 6.2382 15.0333 6.0273 14.5364C5.8084 14.0513 5.3319 13.7337 4.8 13.7182H4.6364C3.7326 13.7182 3 12.9856 3 12.0818C3 11.1781 3.7326 10.4455 4.6364 10.4455H4.7182C5.2697 10.4157 5.7513 10.0625 5.9455 9.5455C6.1564 9.0485 6.0607 8.4744 5.7 8.0727L5.6182 7.9909C5.106 7.6067 4.8648 6.9591 5.0011 6.3335C5.1374 5.7079 5.6261 5.2192 6.2517 5.083C6.8773 4.9467 7.5249 5.1878 7.9091 5.7L7.9909 5.7818C8.3926 6.1425 8.9667 6.2382 9.4636 6.0273H9.5455C10.0305 5.8084 10.3481 5.3319 10.3636 4.8V4.6364C10.3636 3.7326 11.0963 3 12 3C12.9037 3 13.6364 3.7326 13.6364 4.6364V4.7182C13.6519 5.2501 13.9695 5.7266 14.4545 5.9455C14.9515 6.1564 15.5256 6.0607 15.9273 5.7L16.0091 5.6182C16.3933 5.106 17.0409 4.8648 17.6665 5.0011C18.2921 5.1374 18.7808 5.6261 18.917 6.2517C19.0533 6.8773 18.8122 7.5249 18.3 7.9091L18.2182 7.9909C17.8575 8.3926 17.7618 8.9667 17.9727 9.4636L17.9727 9.5455C18.1916 10.0305 18.6681 10.3481 19.2 10.3636H19.3636C20.2674 10.3636 21 11.0963 21 12C21 12.9037 20.2674 13.6364 19.3636 13.6364H19.2818C18.7499 13.6519 18.2734 13.9695 18.0545 14.4545Z"/>',
  chart: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>',
  terminal:
    '<path d="M5 7l4 4-4 4M11 17h8"/><rect x="2.5" y="3.5" width="19" height="17" rx="2"/>',
  layers: '<path d="M12 3l9 5-9 5-9-5 9-5z"/><path d="M3 13l9 5 9-5"/>',
  shading:
    '<circle cx="12" cy="12" r="8"/><path d="M12 4a8 8 0 0 1 0 16z" fill="currentColor" stroke="none" opacity=".5"/>',
  link: '<rect x="4" y="4" width="16" height="16" rx="3"/><circle cx="12" cy="12" r="2.5"/>',
  geom: '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/>',
  light:
    '<path d="M9 18h6M10 21h4M12 3a6 6 0 0 0-3.5 10.9V16h7v-2.1A6 6 0 0 0 12 3z"/>',
  robot:
    '<rect x="5" y="8" width="14" height="11" rx="3"/><path d="M12 4v4M9 13h.01M15 13h.01"/>',
  world:
    '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/>',
  env: '<path d="M3 17c3-5 6-5 9 0s6 5 9 0"/><circle cx="17" cy="7" r="2.5"/>',
  video:
    '<rect x="3" y="6" width="13" height="12" rx="2"/><path d="M16 10l5-3v10l-5-3"/>',
  snap: '<path d="M4 8h3l2-2.5h6L17 8h3v11H4z"/><circle cx="12" cy="13" r="3.5"/>',
  loop: '<path d="M17 2l3 3-3 3"/><path d="M4 11V9a4 4 0 0 1 4-4h12M7 22l-3-3 3-3"/><path d="M20 13v2a4 4 0 0 1-4 4H4"/>',
  follow: '<path d="M5 12h14M13 6l6 6-6 6"/>',
  fit: '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>',
  diamond: '<path d="M12 3l9 9-9 9-9-9z"/>',
  copy: '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
  warn: '<path d="M12 3l10 18H2z"/><path d="M12 10v5M12 18h.01"/>',
  err: '<circle cx="12" cy="12" r="9"/><path d="M9 9l6 6M15 9l-6 6"/>',
  file: '<path d="M14 3H6a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V8z"/><path d="M14 3v5h5"/>',
  upload:
    '<path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>',
  sphere:
    '<circle cx="12" cy="12" r="8"/><ellipse cx="12" cy="12" rx="8" ry="3"/>',
  keyboard:
    '<rect x="2.5" y="6" width="19" height="12" rx="2"/><path d="M6 10h.01M10 10h.01M14 10h.01M18 10h.01M7 14h10"/>',
  mouse: '<rect x="6" y="3" width="12" height="18" rx="6"/><path d="M12 7v4"/>',
  pin: '<path d="M12 17v4M8 3h8l-1 6 3 3v2H6v-2l3-3z"/>',
  check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
  undo: '<path d="M9 14L4 9l5-5"/><path d="M4 9h11a5 5 0 0 1 0 10h-3"/>',
  target:
    '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>',
};
