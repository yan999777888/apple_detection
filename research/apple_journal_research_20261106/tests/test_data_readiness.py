"""Protect leakage reasoning: connected fruit views must stay together."""
import unittest
from apple_research_audit.data_readiness_audit import grouping


def image(name, fruits, split="train", capture=None):
    return {"image_id": name, "split": split, "instances": [
        {"apple_identity": fruit, "capture_group": capture} for fruit in fruits]}


class GroupingTests(unittest.TestCase):
    def test_transitive_covisibility_cannot_be_split_by_one_fruit(self):
        report = grouping([image("a", ["1"]), image("b", ["1", "2"]),
                           image("c", ["2"], "val")], False)
        self.assertEqual(report["component_count"], 1)
        self.assertEqual(report["components_crossing_existing_splits"], 1)
        self.assertEqual(report["components"][0]["unique_fruits"], 2)
        self.assertEqual(report["components"][0]["images"], 3)

    def test_capture_can_link_distinct_fruit_and_identity_namespaces_do_not_collide(self):
        samples = [image("a", ["fruit"], capture="capture"),
                   image("b", ["other"], "val", capture="capture"),
                   image("c", ["capture"], capture="else")]
        self.assertEqual(grouping(samples, False)["component_count"], 3)
        report = grouping(samples, True)
        self.assertEqual(report["component_count"], 2)
        self.assertEqual(report["largest_component_images"], 2)

    def test_empty_images_have_unknown_group_not_one_shared_empty_identity(self):
        report = grouping([image("a", []), image("b", [], "val")], True)
        self.assertEqual(report["component_count"], 2)
        self.assertEqual(report["components_crossing_existing_splits"], 0)
        self.assertTrue(all(r["unique_fruits"] == 0 for r in report["components"]))

    def test_empty_input_does_not_invent_groups(self):
        report = grouping([], True)
        self.assertEqual(report["component_count"], 0)
        self.assertEqual(report["largest_component_fraction"], 0)


if __name__ == "__main__":
    unittest.main()
