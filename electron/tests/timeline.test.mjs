import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'vite';
import { chromium } from 'playwright';
let server, browser, page;
before(async () => {
  server = await createServer({
    configFile: false,
    appType: 'custom',
    root: process.cwd(),
    cacheDir: 'node_modules/.vite-timeline-test',
    server: { host: '127.0.0.1', port: 0, hmr: false },
    optimizeDeps: { include: ['react', 'react-dom/client', 'react/jsx-runtime'] },
  });
  server.middlewares.use('/__timeline', (_, res) => {
    res.setHeader('Content-Type', 'text/html');
    res.end(
      '<!doctype html><base href="/"><link rel="icon" href="data:,"><div id="fixture" style="width:900px;height:260px;display:flex;flex-direction:column"></div>',
    );
  });
  await server.listen();
  browser = await chromium.launch({
    channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome',
    headless: true,
  });
  page = await browser.newPage({ viewport: { width: 1900, height: 650 } });
  await page.goto(`http://127.0.0.1:${server.httpServer.address().port}/__timeline`);
  await page.evaluate(async () => {
    const { default: React } = await import('/node_modules/.vite-timeline-test/deps/react.js');
    const { default: DOM } = await import(
      '/node_modules/.vite-timeline-test/deps/react-dom_client.js'
    );
    const { setLanguage } = await import('/src/i18n.ts');
    const { Group } = await import('/node_modules/three/build/three.module.js');
    setLanguage('en');
    const { Timeline, timelineTimeCaption } = await import('/src/Timeline.tsx');
    await import('/src/design-tokens.css');
    await import('/src/styles.css');
    window.fixture = { begins: 0, commits: 0, cancels: 0, timeCaption: timelineTimeCaption };
    const doc = {
      name: 'Scene',
      root: new Group(),
      joints: Array.from({ length: 9 }, (_, i) => ({
        id: String(i),
        name: `Joint ${i}`,
        type: 'revolute',
      })),
    };
    function Harness() {
      const [keys, setKeys] = React.useState(
        Array.from({ length: 10000 }, (_, i) => ({
          id: String(i),
          time: i / 10,
          pose: { transforms: {}, joints: {}, jointOrigins: {} },
        })),
      );
      const [time, setTime] = React.useState(0);
      const [duration, setDuration] = React.useState(1000);
      const [videoActive, setVideoActive] = React.useState(false);
      const [recording, setRecording] = React.useState(false);
      const [timeline, setTimeline] = React.useState({
        snapshots: [],
        takes: [],
        activeTakeId: null,
      });
      window.fixture.setKeys = setKeys;
      window.fixture.setDuration = setDuration;
      window.fixture.duration = duration;
      window.fixture.time = time;
      window.fixture.setRecording = setRecording;
      window.fixture.setTimeline = setTimeline;
      window.fixture.setTime = setTime;
      window.fixture.setVideoActive = setVideoActive;
      const before = React.useRef({ keys, time, duration });
      window.fixture.keys = keys;
      return React.createElement(Timeline, {
        doc,
        keys,
        time,
        duration,
        playing: false,
        loop: true,
        setLoop: () => {},
        setTime,
        setDuration,
        setPlaying: () => {},
        onCapture: () => {},
        videoActive,
        recording,
        timeline,
        onExportVideo: () => {
          fixture.videoStops = (fixture.videoStops ?? 0) + 1;
          setVideoActive(false);
        },
        onKeys: setKeys,
        onBegin: () => {
          fixture.begins++;
          before.current = { keys, time, duration };
        },
        onCommit: () => fixture.commits++,
        onCancel: () => {
          fixture.cancels++;
          setKeys(before.current.keys);
          setTime(before.current.time);
          setDuration(before.current.duration);
        },
      });
    }
    DOM.createRoot(document.getElementById('fixture')).render(React.createElement(Harness));
  });
  await page.waitForSelector('.keyframe');
  await page.evaluate(() =>
    document.addEventListener(
      'wheel',
      (event) => {
        window.lastWheelX = event.clientX;
      },
      true,
    ),
  );
});
after(async () => {
  await browser?.close();
  await server?.close();
});
test('ten thousand keys retain bounded visible DOM and reveal actual keys after scrolling', async () => {
  assert.equal(await page.locator('.key-lane').count(), 1);
  assert.equal(await page.locator('.lane-label').count(), 3);
  assert.ok((await page.locator('.keyframe').count()) < 100);
  assert.ok((await page.locator('.ruler-tick').count()) < 20);
  await scrollToTime(500);
  await page.waitForFunction(() =>
    [...document.querySelectorAll('.keyframe')].some((el) =>
      el.getAttribute('aria-label').includes('500.00'),
    ),
  );
  assert.ok((await page.locator('.keyframe').count()) < 100);
  assert.ok((await page.locator('.ruler-tick').count()) < 20);
  assert.equal(await page.evaluate(() => fixture.keys.length), 10000);
});
test('Escape during held keyframe drag restores data and releases ownership before the next gesture', async () => {
  await scrollToTime(0);
  await page.waitForTimeout(100);
  let marker = page.getByRole('button', { name: 'Keyframe 0.30 seconds Keyframes', exact: true });
  let rect = await marker.boundingBox();
  await page.mouse.move(rect.x + rect.width / 2, rect.y + rect.height / 2);
  await page.mouse.down();
  await page.mouse.move(rect.x + 100, rect.y + rect.height / 2, { steps: 6 });
  await page.keyboard.press('Escape');
  await page.mouse.up();
  assert.equal(await page.evaluate(() => fixture.keys.find((key) => key.id === '3').time), 0.3);
  assert.equal(await page.evaluate(() => fixture.cancels), 1);
  assert.equal(await page.evaluate(() => fixture.commits), 0);
  marker = page.getByRole('button', { name: 'Keyframe 0.30 seconds Keyframes', exact: true });
  rect = await marker.boundingBox();
  await page.mouse.move(rect.x + rect.width / 2, rect.y + rect.height / 2);
  await page.mouse.down();
  await page.mouse.move(rect.x + rect.width / 2 + 88, rect.y + rect.height / 2, { steps: 6 });
  await page.mouse.up();
  assert.equal(await page.evaluate(() => fixture.keys.find((key) => key.id === '3').time), 1.3);
  assert.equal(await page.evaluate(() => fixture.commits), 1);
});

test('active video export reveals Stop beside Record without hiding Follow in a narrow toolbar', async () => {
  await page.locator('#fixture').evaluate((el) => {
    el.style.width = '480px';
  });
  await page.evaluate(() => fixture.setVideoActive(true));
  const stop = page
    .locator('.timeline-toolbar')
    .getByRole('button', { name: 'Stop video', exact: true });
  await stop.waitFor();
  const metrics = await page.locator('.timeline-toolbar').evaluate((el) => {
    const rect = (n) => {
      const b = n.getBoundingClientRect();
      return { left: b.left, right: b.right, top: b.top };
    };
    return {
      bar: rect(el),
      stop: rect(el.querySelector('.timeline-video-button')),
      follow: rect(el.querySelector('.timeline-follow')),
      range: rect([...el.querySelectorAll('.timeline-range-fields')].find((n) => n.offsetWidth)),
    };
  });
  for (const key of ['stop', 'follow', 'range'])
    assert.ok(
      metrics[key].left >= metrics.bar.left && metrics[key].right <= metrics.bar.right,
      `${key} stays visible`,
    );
  assert.equal(
    await stop.evaluate((el) => el.previousElementSibling.classList.contains('record-button')),
    true,
  );
  assert.equal(await page.locator('.timeline-actions-popup').count(), 0);
  await stop.click();
  assert.equal(await page.evaluate(() => fixture.videoStops), 1);
  assert.equal(await stop.count(), 0);
});

test('inline range editors retain fixed frames as recorded timestamps gain integer digits', async () => {
  await page.locator('#fixture').evaluate((el) => {
    el.style.width = '1000px';
  });
  await page.setViewportSize({ width: 1100, height: 460 });
  const rects = () =>
    page.locator('.timeline-toolbar').evaluate((el) => {
      const rect = (n) => {
        const b = n.getBoundingClientRect();
        return [b.x, b.y, b.width, b.height];
      };
      return [
        ...el.querySelectorAll(
          '.timeline-follow,.timeline-range-fields,.timeline-range-fields .number-field,.timeline-video-button,.timeline-primary-controls',
        ),
      ]
        .filter((n) => n.offsetWidth)
        .map(rect);
    });
  await page.getByRole('textbox', { name: 'View start', exact: true }).waitFor();
  const baseline = await rects();
  for (const end of [0, 9.999, 10, 100]) {
    await page.evaluate((end) => {
      const pose = { transforms: {}, joints: {}, jointOrigins: {} };
      fixture.setTime(end);
      fixture.setRecording(true);
      fixture.setTimeline({
        snapshots: [],
        activeTakeId: 'recording',
        takes: [
          {
            id: 'recording',
            name: 'Take',
            frames: [
              { time: 0, pose },
              { time: end, pose },
            ],
          },
        ],
      });
    }, end);
    const startField = page.getByRole('textbox', { name: 'Loop start', exact: true });
    const endField = page.getByRole('textbox', { name: 'Loop end', exact: true });
    await page.waitForFunction(
      (end) => document.querySelector('[aria-label="Loop end"]')?.value === end.toFixed(3),
      end,
    );
    assert.equal(await startField.inputValue(), '0.000');
    assert.equal(await endField.inputValue(), end.toFixed(3));
    assert.equal(await endField.isDisabled(), true, 'Recording owns the growing range');
    assert.equal(
      await endField.evaluate(
        (el) => !!el.closest('.timeline-toolbar') && !el.closest('[popover]'),
      ),
      true,
    );
    const current = await rects();
    assert.deepEqual(current, baseline, `Recorded endpoint ${end} cannot move adjacent controls`);
  }
  await page.evaluate(() => fixture.setRecording(false));
  assert.equal(
    await page.getByRole('textbox', { name: 'Loop end', exact: true }).isDisabled(),
    false,
  );
  assert.equal(await page.locator('.timeline-range-fields button').count(), 0);
});

async function editNumber(label, value) {
  const input = page.getByRole('textbox', { name: label, exact: true });
  await input.dblclick();
  await input.fill(String(value));
  await input.press('Enter');
}
async function resetExtent(keys = []) {
  await page.setViewportSize({ width: 1900, height: 650 });
  await page.locator('#fixture').evaluate((el) => {
    el.style.width = '1400px';
  });
  await page.evaluate((keys) => {
    fixture.setRecording(false);
    fixture.setTime(0);
    fixture.setDuration(10);
    fixture.setKeys(
      keys.map((key) => ({ ...key, pose: { transforms: {}, joints: {}, jointOrigins: {} } })),
    );
    fixture.setTimeline({ snapshots: [], takes: [], activeTakeId: null });
  }, keys);
  await page.getByRole('textbox', { name: 'View end', exact: true }).waitFor();
  await page.getByRole('button', { name: 'Fit timeline', exact: true }).click();
  await editNumber('Timeline zoom', 100);
  await scrollToTime(0);
}
async function projection() {
  return page.locator('.ruler-tick').evaluateAll((ticks) => {
    const [a, b] = ticks;
    const ta = parseFloat(a.textContent),
      tb = parseFloat(b.textContent);
    const ax = a.getBoundingClientRect().x,
      bx = b.getBoundingClientRect().x;
    const pixelsPerSecond = (bx - ax) / (tb - ta);
    return { pixelsPerSecond, origin: ax - ta * pixelsPerSecond };
  });
}

test('time edits extend content while manual view navigation stays independent of recording', async () => {
  await resetExtent();
  assert.equal(await page.getByRole('textbox', { name: 'Timeline end', exact: true }).count(), 0);
  assert.equal(await page.locator('.timeline-duration').count(), 0);
  const before = await projection();
  await editNumber('Current time', 12.5);
  assert.equal(await page.evaluate(() => fixture.time), 12.5);
  assert.equal(await page.evaluate(() => fixture.duration), 20);
  assert.deepEqual(await projection(), before, 'Content extension cannot rescale the current view');
  await editNumber('Current time', 3700.25);
  assert.equal(await page.evaluate(() => fixture.time), 3700.25);
  assert.equal(await page.evaluate(() => fixture.duration), 3710);
  assert.deepEqual(await projection(), before);
  await editNumber('View end', 4000);
  await editNumber('View start', 3900);
  await page.waitForFunction(
    () => Number(document.querySelector('[aria-label="View start"]').value) === 3900,
  );
  assert.equal(
    await page.evaluate(() => fixture.duration),
    3710,
    'View navigation cannot author content',
  );
  assert.equal(
    Number(await page.getByRole('textbox', { name: 'View start', exact: true }).inputValue()),
    3900,
  );
  assert.equal(
    Number(await page.getByRole('textbox', { name: 'View end', exact: true }).inputValue()),
    4000,
  );
});

test('key drag, nudge and metadata extend the timeline instead of clamping to its former extent', async () => {
  await resetExtent([{ id: 'growing', name: 'Growing key', time: 9 }]);
  const marker = page.locator('[data-scene-frame="growing"]');
  await marker.waitFor();
  const box = await marker.boundingBox(),
    scale = await projection();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 3 * scale.pixelsPerSecond, box.y + box.height / 2, {
    steps: 5,
  });
  await page.mouse.up();
  assert.equal(await page.evaluate(() => fixture.keys[0].time), 12);
  assert.equal(await page.evaluate(() => fixture.duration), 20);
  assert.deepEqual(
    await projection(),
    scale,
    'Moving a key past the old extent preserves projection',
  );

  await resetExtent([{ id: 'edge', name: 'Edge key', time: 10 }]);
  await page.locator('[data-scene-frame="edge"]').click();
  await page.getByTestId('timeline').press('ArrowRight');
  assert.equal(await page.evaluate(() => fixture.keys[0].time), 10 + 1 / 60);
  assert.equal(await page.evaluate(() => fixture.duration), 20);
  await editNumber('Keyframe time', 3700.25);
  await page.getByRole('button', { name: 'Apply keyframe details', exact: true }).click();
  assert.equal(await page.evaluate(() => fixture.keys[0].time), 3700.25);
  assert.equal(await page.evaluate(() => fixture.duration), 3710);
});

test('content growth during recording preserves pixels per second and sample loop endpoints', async () => {
  await resetExtent();
  const before = await projection();
  await page.getByRole('button', { name: 'Follow off', exact: true }).click();
  for (const end of [9.99, 10.01, 20.01, 3700.25]) {
    await page.evaluate((end) => {
      const pose = { transforms: {}, joints: {}, jointOrigins: {} };
      fixture.setRecording(true);
      fixture.setTime(end);
      fixture.setDuration(Math.ceil(end / 10) * 10);
      fixture.setTimeline({
        snapshots: [],
        activeTakeId: 'long',
        takes: [
          {
            id: 'long',
            name: 'Long take',
            frames: [
              { time: 0, pose },
              { time: end, pose },
            ],
          },
        ],
      });
    }, end);
    await page.waitForFunction((end) => fixture.time === end, end);
    assert.deepEqual(await projection(), before);
    assert.equal(
      Number(await page.getByRole('textbox', { name: 'Loop end', exact: true }).inputValue()),
      end,
    );
    assert.equal(
      await page.getByRole('textbox', { name: 'Loop end', exact: true }).isDisabled(),
      true,
    );
  }
  await page.evaluate(() => fixture.setRecording(false));
  assert.equal(
    await page.getByRole('textbox', { name: 'Loop end', exact: true }).isDisabled(),
    false,
  );
});

test('long time captions remain within eight characters without changing the stored value or text precision', async () => {
  await resetExtent();
  const samples = [
    0,
    9.999,
    10,
    100,
    9999.999,
    10000.125,
    10000000,
    100000000,
    1e100,
    Number.MAX_VALUE,
  ];
  const captions = await page.evaluate(
    (values) => values.map((value) => fixture.timeCaption(value)),
    samples,
  );
  for (let i = 0; i < samples.length; i++) {
    assert.ok(captions[i].length <= 8, `${samples[i]}: ${captions[i]}`);
    assert.ok(
      Number.isFinite(Number(captions[i])),
      'The idle caption must be a complete numeric representation',
    );
  }
  assert.deepEqual(captions.slice(0, 5), ['0.000', '9.999', '10.000', '100.000', '9999.999']);
  await editNumber('Current time', 10000.125);
  const field = page.getByRole('textbox', { name: 'Current time', exact: true });
  assert.equal(await field.inputValue(), '10000.13');
  assert.equal(await page.evaluate(() => fixture.time), 10000.125);
  await field.dblclick();
  assert.equal(await field.inputValue(), '10000.125');
  await field.press('Enter');
  assert.equal(await page.evaluate(() => fixture.time), 10000.125);
});

test('Follow preserves scale, playhead fraction, and projection after Stop', async () => {
  await resetExtent();
  const initial = await projection();
  await page.getByRole('button', { name: 'Follow page', exact: true }).click();
  await page.evaluate(() => {
    fixture.setTime(20);
    fixture.setDuration(20);
    fixture.setRecording(true);
  });
  await page.waitForFunction(
    () => Number(document.querySelector('.timeline-body').dataset.viewStart) > 0,
  );
  let range = await view();
  const anchor = (20 - range.start) / (range.end - range.start);
  assert.ok(Math.abs(anchor - 0.08) < 1e-8);
  assert.ok(
    Math.abs((await projection()).pixelsPerSecond - initial.pixelsPerSecond) < 0.03,
    JSON.stringify({ initial, after: await projection(), range }),
  );
  await page.getByRole('button', { name: 'Follow locked', exact: true }).click();
  await page.evaluate(() => {
    fixture.setTime(40);
    fixture.setDuration(40);
  });
  await page.waitForFunction(
    () => Number(document.querySelector('.timeline-body').dataset.viewStart) > 30,
  );
  range = await view();
  assert.ok(Math.abs((40 - range.start) / (range.end - range.start) - anchor) < 1e-8);
  const beforeStop = await projection();
  await page.evaluate(() => fixture.setRecording(false));
  await page.waitForFunction(() => !document.querySelector('[aria-label="Current time"]').disabled);
  assert.deepEqual(await projection(), beforeStop);
  assert.deepEqual(await view(), range);
});

test('dense takes visit only their visible samples while scrolling long recordings', async () => {
  await resetExtent();
  await page.getByRole('button', { name: 'Follow off', exact: true }).click();
  await page.evaluate(() => {
    fixture.frameReads = 0;
    const pose = { transforms: {}, joints: {}, jointOrigins: {} };
    const frames = Array.from({ length: 100000 }, (_, index) => ({
      get time() {
        fixture.frameReads++;
        return index;
      },
      pose,
    }));
    fixture.setDuration(100000);
    fixture.setTimeline({
      snapshots: [],
      activeTakeId: 'dense',
      takes: [{ id: 'dense', name: 'Dense', frames }],
    });
  });
  await page.getByRole('textbox', { name: 'Loop end', exact: true }).waitFor();
  assert.ok(
    await page.evaluate(() => fixture.frameReads < 1000),
    'Initial drawing must not scan historical samples',
  );
  await page.evaluate(() => {
    fixture.frameReads = 0;
  });
  await scrollToTime(50000);
  await page.waitForFunction(() => fixture.frameReads > 0);
  assert.ok(
    await page.evaluate(() => fixture.frameReads < 1000),
    'Scrolling uses a chronological lower bound then the visible sample interval',
  );
});

async function view() {
  return page
    .locator('.timeline-body')
    .evaluate((el) => ({ start: Number(el.dataset.viewStart), end: Number(el.dataset.viewEnd) }));
}
async function scrollToTime(time) {
  await page.locator('.timeline-body').evaluate((el, time) => {
    el.scrollLeft =
      ((time - Number(el.dataset.scrollLower)) / Number(el.dataset.scrollSpan)) *
      Number(el.dataset.scrollTravel);
  }, time);
  await page.waitForFunction(
    (time) =>
      Math.abs(Number(document.querySelector('.timeline-body').dataset.viewStart) - time) <
      Math.max(
        0.1,
        Number(document.querySelector('.timeline-body').dataset.scrollSpan) /
          Number(document.querySelector('.timeline-body').dataset.scrollTravel),
      ),
    time,
  );
}
async function wheelAt(ratio, deltaY, deltaMode = 0, deltaX = 0) {
  const box = await page.locator('.timeline-track').boundingBox();
  const scale = await page
    .locator('.timeline-track')
    .evaluate((el) => el.getBoundingClientRect().width / el.offsetWidth);
  const x = box.x + 12 * scale + ratio * (box.width - 24 * scale),
    y = box.y + 60 * scale;
  await page.mouse.move(x, y);
  if (deltaMode === 0 && !deltaX) await page.mouse.wheel(0, deltaY);
  else
    await page
      .locator('.timeline-track')
      .dispatchEvent('wheel', { clientX: x, clientY: y, deltaY, deltaX, deltaMode });
  await page.waitForTimeout(30);
  return page.locator('.timeline-track').evaluate((el) => {
    const b = el.getBoundingClientRect(),
      scale = b.width / el.offsetWidth;
    return (window.lastWheelX - b.left - 12 * scale) / (b.width - 24 * scale);
  });
}

test('wheel keeps the cursor time fixed across zero at normal and fractional UI scale', async () => {
  for (const scale of [1, 1.25]) {
    await resetExtent();
    await page.locator('#fixture').evaluate((el, scale) => (el.style.zoom = scale), scale);
    await page.getByRole('button', { name: 'Follow page', exact: true }).click();
    await page.waitForTimeout(50);
    for (const ratio of [0.08, 0.35, 0.9]) {
      for (const delta of [100, -100, 2, -2]) {
        const before = await view(),
          span = before.end - before.start;
        const actualRatio = await wheelAt(ratio, delta);
        const anchor = before.start + actualRatio * span;
        const after = await view(),
          nextSpan = after.end - after.start;
        assert.ok(
          Math.abs(anchor - (after.start + actualRatio * nextSpan)) < 1e-8,
          `${scale}/${ratio}/${delta}: pointer drift ${JSON.stringify({ before, after, anchor, geometry: await page.locator('.timeline-track').evaluate((el) => ({ rect: el.getBoundingClientRect().toJSON(), width: el.offsetWidth, parent: el.parentElement.getBoundingClientRect().toJSON() })) })}`,
        );
        assert.ok(
          Math.abs(nextSpan / span - Math.exp((delta / 100) * 0.18)) < 1e-8,
          JSON.stringify({ scale, ratio, delta, before, after }),
        );
      }
    }
    assert.equal(await page.evaluate(() => fixture.duration), 10);
    await page.screenshot({
      path: `../output/electron/verification/timeline-wheel-anchor-${scale}x.png`,
    });
  }
  await page.locator('#fixture').evaluate((el) => (el.style.zoom = 1));
});

test('batched wheel input saturates, reverses immediately and never changes recording extent', async () => {
  await resetExtent();
  await page.getByRole('button', { name: 'Follow off', exact: true }).click();
  const range = await view(),
    ratio = 0.35;
  let anchor;
  const burst = async (deltaY) => {
    const actualRatio = await page.locator('.timeline-track').evaluate(
      (el, { deltaY, ratio }) => {
        const b = el.getBoundingClientRect(),
          scale = b.width / el.offsetWidth;
        const clientX = b.x + scale * 12 + ratio * (b.width - scale * 24);
        const event = new WheelEvent('wheel', {
          deltaY,
          clientX,
          clientY: b.y + 60,
          bubbles: true,
          cancelable: true,
        });
        const actualRatio = (event.clientX - b.x - scale * 12) / (b.width - scale * 24);
        for (let i = 0; i < 150; i++)
          el.dispatchEvent(
            new WheelEvent('wheel', {
              deltaY,
              clientX,
              clientY: b.y + 60,
              bubbles: true,
              cancelable: true,
            }),
          );
        return actualRatio;
      },
      { deltaY, ratio },
    );
    await page.waitForTimeout(50);
    return actualRatio;
  };
  const actualRatio = await burst(1000000);
  anchor = range.start + actualRatio * (range.end - range.start);
  let current = await view();
  assert.ok(Math.abs(current.end - current.start - 60) < 1e-8);
  assert.ok(
    Math.abs(current.start + actualRatio * 60 - anchor) < 1e-8,
    JSON.stringify({ current, range, anchor }),
  );
  await wheelAt(ratio, -100);
  current = await view();
  assert.ok(current.end - current.start < 60);
  await burst(-1000000);
  current = await view();
  assert.ok(Math.abs(current.end - current.start - 0.001) < 1e-9);
  assert.ok(Math.abs(current.start + actualRatio * 0.001 - anchor) < 1e-8);
  const trackWidth = await page.locator('.timeline-track').evaluate((el) => el.offsetWidth);
  assert.ok(trackWidth < 2000, 'Rendering stays viewport-sized at maximum zoom');
  assert.ok(await page.locator('.timeline-body').evaluate((el) => el.scrollWidth < 1002000));
  await wheelAt(ratio, 100);
  current = await view();
  assert.ok(current.end - current.start > 0.001);
  assert.equal(await page.evaluate(() => fixture.duration), 10);
});

test('pixel, line and page wheel units agree; labels and numeric fields do not zoom tracks', async () => {
  for (const [delta, mode] of [
    [100, 0],
    [3, 1],
    [1, 2],
  ]) {
    await resetExtent();
    const before = await view();
    await wheelAt(0.5, delta, mode);
    const after = await view();
    assert.ok(
      Math.abs((after.end - after.start) / (before.end - before.start) - Math.exp(0.18)) < 1e-8,
    );
  }
  const before = await view();
  await page
    .locator('.lane-header')
    .dispatchEvent('wheel', { deltaY: 100, clientX: 40, clientY: 180 });
  await page
    .getByRole('textbox', { name: 'Timeline zoom', exact: true })
    .dispatchEvent('wheel', { deltaY: 100 });
  assert.deepEqual(await view(), before);
});
