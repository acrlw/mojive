export type ResourceFile = Pick<File, 'name' | 'size' | 'webkitRelativePath' | 'arrayBuffer'>;

/** Read every directory batch and retain paths; never silently replace duplicate basenames. */
export async function droppedFiles(transfer: DataTransfer): Promise<ResourceFile[]> {
  const entries = Array.from(transfer.items).filter((item) => item.kind === 'file')
    .map((item) => ({ entry: item.webkitGetAsEntry?.(), file: item.getAsFile() }));
  const result: ResourceFile[] = [];
  let size = 0;
  const add = (file: File, path: string) => {
    size += file.size;
    if (result.length >= 10000 || size > 512 * 1024 * 1024)
      throw new Error('Model resources exceed the package limit (10,000 files or 512 MiB).');
    result.push({ name: file.name, size: file.size, webkitRelativePath: path,
      arrayBuffer: () => file.arrayBuffer() });
  };
  const visit = async (entry: FileSystemEntry): Promise<void> => {
    if (entry.isFile) {
      const file = await new Promise<File>((resolve, reject) => (entry as FileSystemFileEntry).file(resolve, reject));
      add(file, entry.fullPath.replace(/^\//, ''));
    } else if (entry.isDirectory) {
      const reader = (entry as FileSystemDirectoryEntry).createReader();
      for (;;) {
        const batch = await new Promise<FileSystemEntry[]>((resolve, reject) => reader.readEntries(resolve, reject));
        if (!batch.length) break;
        for (const child of batch) await visit(child);
      }
    }
  };
  if (entries.length) {
    for (const item of entries) {
      if (item.entry) await visit(item.entry);
      else if (item.file) add(item.file, item.file.webkitRelativePath || item.file.name);
    }
  } else for (const file of Array.from(transfer.files)) add(file, file.webkitRelativePath || file.name);
  return result;
}
