import { readFile, realpath } from 'node:fs/promises';
import path from 'node:path';

// Model filenames are opaque data, not Vite module URLs. Its /@fs middleware can
// double-encode reserved characters such as commas and return the HTML fallback.
export async function routeReviewResources(page, directory) {
  const root = await realpath(directory);
  await page.route('**/__model_resource?*', async (route) => {
    const resource = new URL(route.request().url()).searchParams.get('path');
    if (!resource) return route.fulfill({ status: 400, body: 'Missing resource path' });
    try {
      const file = await realpath(path.resolve(root, resource));
      const relative = path.relative(root, file);
      if (relative === '..' || relative.startsWith(`..${path.sep}`) || path.isAbsolute(relative)) {
        return route.fulfill({ status: 403, body: 'Resource outside review directory' });
      }
      await route.fulfill({ contentType: 'application/octet-stream', body: await readFile(file) });
    } catch (error) {
      await route.fulfill({ status: error.code === 'ENOENT' ? 404 : 500, body: String(error) });
    }
  });
}
