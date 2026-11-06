/** Load an explicitly editable scene; startup is a live MuJoCo model. */
export async function openEditableScene(page, chinese = false) {
  await page.getByRole('treeitem').first().waitFor();
  await page.getByRole('navigation', { name: chinese ? '应用菜单' : 'Application menu' })
    .getByRole('button', { name: chinese ? '文件' : 'File', exact: true }).click();
  await page.getByRole('menuitem', { name: chinese ? '场景对象' : 'Scene objects', exact: true }).click();
  await page.waitForFunction(() => document.querySelector('.document-title')?.textContent?.match(/Scene objects|场景对象/));
  await page.locator('.busy-overlay').waitFor({ state: 'detached' });
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
