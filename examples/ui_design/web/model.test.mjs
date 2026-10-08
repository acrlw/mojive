import assert from "node:assert/strict";
import {
  History,
  initialDocument,
  clone,
  samplePose,
  numericValue,
  uniqueName,
  upsertKey,
} from "./model.mjs";
let passed = 0;
function check(name, fn) {
  fn();
  passed++;
  console.log("PASS " + name);
}
check("History commits atomically and rejects a failed transaction", () => {
  const h = new History(initialDocument()),
    before = clone(h.document);
  assert.throws(() =>
    h.commit("invalid", (d) => {
      d.entities = [];
      throw Error("fail");
    }),
  );
  assert.deepEqual(h.document, before);
  assert.equal(h.past.length, 0);
});
check("Undo and redo preserve structural edits and material data", () => {
  const h = new History(initialDocument());
  const removed = clone(h.document.entities[0]);
  h.commit("remove", (d) => d.entities.shift());
  h.undo();
  assert.deepEqual(h.document.entities[0], removed);
  h.redo();
  assert.equal(h.document.entities.length, 7);
});
check(
  "New edits invalidate redo history, no-op edits do not create entries",
  () => {
    const h = new History(initialDocument());
    assert.equal(
      h.commit("same", (d) => (d.name = "Joint study")),
      false,
    );
    h.commit("rename", (d) => (d.name = "a"));
    h.undo();
    h.commit("rename", (d) => (d.name = "b"));
    assert.equal(h.future.length, 0);
  },
);
check("Save baseline remains accurate through undo and redo", () => {
  const h = new History(initialDocument());
  h.commit("rename", (d) => (d.name = "a"));
  assert.equal(h.dirty, true);
  h.markSaved();
  assert.equal(h.dirty, false);
  h.undo();
  assert.equal(h.dirty, true);
  h.redo();
  assert.equal(h.dirty, false);
});
check("Interpolation is deterministic, clamped and non-mutating", () => {
  const d = initialDocument(),
    keys = clone(d.keys);
  assert.deepEqual(samplePose(d.keys, 2), {
    hinge: -17.5,
    ball: 10,
    slide: 0.175,
  });
  assert.deepEqual(samplePose(d.keys, -1), d.keys[0].pose);
  assert.deepEqual(samplePose(d.keys, 11), d.keys[2].pose);
  assert.deepEqual(d.keys, keys);
  assert.equal(samplePose([], 4), null);
});
check("Capturing near an existing time replaces only that pose", () => {
  const d = initialDocument();
  d.pose.hinge = 42;
  const id = upsertKey(d, 4.01, "new");
  assert.equal(id, "key-1");
  assert.equal(d.keys.length, 3);
  assert.equal(d.keys[1].pose.hinge, 42);
  upsertKey(d, 3, "next");
  assert.equal(d.keys.length, 4);
  assert.equal(d.keys[1].time, 3);
});
check(
  "Number validation rejects empty, nonfinite and out-of-range input",
  () => {
    for (const v of ["", "NaN", Infinity, -1, 101])
      assert.throws(() => numericValue(v, { min: 0.01, max: 100 }));
    assert.equal(numericValue(".125", { min: 0.01, max: 100 }), 0.125);
  },
);
check("Names reject duplicates, whitespace and oversized values", () => {
  const d = initialDocument();
  for (const n of ["", "  ", "FREE_BODY", "x".repeat(81)])
    assert.throws(() => uniqueName(d.entities, n));
  assert.equal(uniqueName(d.entities, "free_body", "free_body"), "free_body");
});
const { groupKeys } = await import("./model.mjs");
check(
  "Dense timeline keys share a chooser and retain every key identity",
  () => {
    const keys = [
      { id: "a", time: 1 },
      { id: "b", time: 1.1 },
      { id: "c", time: 5 },
    ];
    const groups = groupKeys(keys, 300, 10, 34);
    assert.deepEqual(
      groups.map((g) => g.map((k) => k.id)),
      [["a", "b"], ["c"]],
    );
    assert.equal(groupKeys(keys, 4000, 10, 34).length, 3);
  },
);
console.log(`Final: ${passed} model checks passed`);
