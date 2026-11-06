import { readFileSync } from 'node:fs';

const nativeChinese = JSON.parse(readFileSync(new URL('../../src/locales/zh.json', import.meta.url), 'utf8'));
const electronChinese = JSON.parse(readFileSync(new URL('../../src/locales/electron-zh.json', import.meta.url), 'utf8'));
const chineseCopy = { ...nativeChinese, ...electronChinese };
const label = (text, chinese) => chinese ? chineseCopy[text] ?? text : text;

/** Use the asset browser, the production owner for sample scenes and previews.
 * Return before confirmation so callers can exercise unsaved-document protection.
 */
export async function chooseSampleScene(page, name, chinese = false) {
  await page.locator('.busy-overlay').waitFor({ state: 'detached' });
  await page.locator('.activity-rail').getByRole('button', { name: label('Assets', chinese), exact: true }).click();
  await page.locator('.asset-browser').getByRole('button', { name: label('Samples', chinese), exact: true }).click();
  await page.locator('.asset-row').filter({ has: page.locator('strong', { hasText: label(name, chinese) }) })
    .getByRole('button', { name: label('Open model', chinese), exact: true }).click();
}

export async function openSampleScene(page, name, chinese = false) {
  await chooseSampleScene(page, name, chinese);
  await page.waitForFunction((name) => document.querySelector('.document-title')?.textContent?.includes(name), label(name, chinese));
  await page.locator('.busy-overlay').waitFor({ state: 'detached' });
  await page.locator('.activity-rail').getByRole('button', { name: label('Scene', chinese), exact: true }).click();
}

/** Load an explicitly editable scene; startup is a live MuJoCo model. */
export async function openEditableScene(page, chinese = false) {
  await page.getByRole('treeitem').first().waitFor();
  await openSampleScene(page, 'Scene objects', chinese);
}

/** Open the maintained kinematic joint fixture through the workspace import path.
 * Tests of local authoring must not depend on the live physics startup model.
 */
export async function openEditableJointScene(page) {
  await page.getByRole('treeitem').first().waitFor();
  await page.locator('.busy-overlay').waitFor({ state: 'detached' });
  await page.evaluate(async () => {
    const { createCatalogScene } = await import('/src/SceneCatalog.ts');
    const { saveWorkspace } = await import('/src/editor.ts');
    const doc = await createCatalogScene('joint_gizmo');
    const transfer = new DataTransfer();
    transfer.items.add(new File([saveWorkspace(doc, [], 10)], 'joint-fixture.mojive.json', {
      type: 'application/json',
    }));
    document.querySelector('.app-shell').dispatchEvent(new DragEvent('drop', {
      bubbles: true, cancelable: true, dataTransfer: transfer,
    }));
  });
  await page.locator('.tree-row.type-robot[data-tooltip="joint_gizmo_acceptance"]').waitFor();
  await page.locator('.busy-overlay').waitFor({ state: 'detached' });
}
