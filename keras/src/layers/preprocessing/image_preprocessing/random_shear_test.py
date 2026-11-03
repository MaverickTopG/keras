import numpy as np
from absl.testing import parameterized
from tensorflow import data as tf_data

import keras
from keras.src import backend
from keras.src import layers
from keras.src import testing
from keras.src.utils import backend_utils


class RandomShearTest(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            layers.RandomShear,
            init_kwargs={
                "x_factor": (0.5, 1),
                "y_factor": (0.5, 1),
                "interpolation": "bilinear",
                "fill_mode": "reflect",
                "data_format": "channels_last",
                "seed": 1,
            },
            input_shape=(8, 3, 4, 3),
            supports_masking=False,
            expected_output_shape=(8, 3, 4, 3),
        )

    def test_random_posterization_inference(self):
        seed = 3481
        layer = layers.RandomShear(1, 1)
        np.random.seed(seed)
        inputs = np.random.randint(0, 255, size=(224, 224, 3))
        output = layer(inputs, training=False)
        self.assertAllClose(inputs, output)

    def test_shear_pixel_level(self):
        image = np.zeros((1, 5, 5, 3))
        image[0, 1:4, 1:4, :] = 1.0
        image[0, 2, 2, :] = [0.0, 1.0, 0.0]
        image = keras.ops.convert_to_tensor(image, dtype="float32")

        data_format = backend.config.image_data_format()
        if data_format == "channels_first":
            image = keras.ops.transpose(image, (0, 3, 1, 2))

        shear_layer = layers.RandomShear(
            x_factor=(0.2, 0.3),
            y_factor=(0.2, 0.3),
            interpolation="bilinear",
            fill_mode="constant",
            fill_value=0.0,
            seed=42,
            data_format=data_format,
        )

        sheared_image = shear_layer(image)

        if data_format == "channels_first":
            sheared_image = keras.ops.transpose(sheared_image, (0, 2, 3, 1))

        original_pixel = image[0, 2, 2, :]
        sheared_pixel = sheared_image[0, 2, 2, :]
        self.assertNotAllClose(original_pixel, sheared_pixel)

    def test_tf_data_compatibility(self):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            input_data = np.random.random((2, 8, 8, 3))
        else:
            input_data = np.random.random((2, 3, 8, 8))
        layer = layers.RandomShear(1, 1)

        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            output.numpy()

    @parameterized.named_parameters(
        (
            "with_x_shift",
            [[1.0, 0.0]],
            [[[0.0, 1.0, 3.5, 3.0], [0.5, 4.0, 4.5, 6.0]]],
        ),
        (
            "with_y_shift",
            [[0.0, 1.0]],
            [[[2.0, 0.0, 4.0, 1.5], [6.0, 0.0, 8.0, 0.5]]],
        ),
        (
            "with_xy_shift",
            [[0.5, 0.5]],
            [[[5 / 6, 0.0, 29 / 6, 17 / 6], [25 / 6, 1 / 6, 8.0, 25 / 6]]],
        ),
    )
    def test_random_shear_bounding_boxes(self, translation, expected_boxes):
        data_format = backend.config.image_data_format()
        if data_format == "channels_last":
            image_shape = (10, 8, 3)
        else:
            image_shape = (3, 10, 8)
        input_image = np.random.random(image_shape)
        bounding_boxes = {
            "boxes": np.array(
                [
                    [2, 1, 4, 3],
                    [6, 4, 8, 6],
                ]
            ),
            "labels": np.array([[1, 2]]),
        }
        input_data = {"images": input_image, "bounding_boxes": bounding_boxes}
        layer = layers.RandomShear(
            x_factor=0.5,
            y_factor=0.5,
            data_format=data_format,
            seed=42,
            bounding_box_format="xyxy",
        )

        transformation = {
            "shear_factor": backend_utils.convert_tf_tensor(
                np.array(translation)
            ),
            "input_shape": image_shape,
        }
        output = layer.transform_bounding_boxes(
            input_data["bounding_boxes"],
            transformation=transformation,
            training=True,
        )

        self.assertAllClose(output["boxes"], expected_boxes)

    @parameterized.named_parameters(
        (
            "with_x_shift",
            [[1.0, 0.0]],
            [[[0.0, 1.0, 3.5, 3.0], [0.5, 4.0, 4.5, 6.0]]],
        ),
        (
            "with_y_shift",
            [[0.0, 1.0]],
            [[[2.0, 0.0, 4.0, 1.5], [6.0, 0.0, 8.0, 0.5]]],
        ),
        (
            "with_xy_shift",
            [[0.5, 0.5]],
            [[[5 / 6, 0.0, 29 / 6, 17 / 6], [25 / 6, 1 / 6, 8.0, 25 / 6]]],
        ),
    )
    def test_random_shear_tf_data_bounding_boxes(
        self, translation, expected_boxes
    ):
        data_format = backend.config.image_data_format()
        if backend.config.image_data_format() == "channels_last":
            image_shape = (1, 10, 8, 3)
        else:
            image_shape = (1, 3, 10, 8)
        input_image = np.random.random(image_shape)
        bounding_boxes = {
            "boxes": np.array(
                [
                    [
                        [2, 1, 4, 3],
                        [6, 4, 8, 6],
                    ]
                ]
            ),
            "labels": np.array([[1, 2]]),
        }

        input_data = {"images": input_image, "bounding_boxes": bounding_boxes}

        ds = tf_data.Dataset.from_tensor_slices(input_data)
        layer = layers.RandomShear(
            x_factor=0.5,
            y_factor=0.5,
            data_format=data_format,
            seed=42,
            bounding_box_format="xyxy",
        )

        transformation = {
            "shear_factor": np.array(translation),
            "input_shape": image_shape,
        }

        ds = ds.map(
            lambda x: layer.transform_bounding_boxes(
                x["bounding_boxes"],
                transformation=transformation,
                training=True,
            )
        )

        output = next(iter(ds))
        expected_boxes = np.array(expected_boxes)
        self.assertAllClose(output["boxes"], expected_boxes)

    @parameterized.named_parameters(
        ("negative", [[-0.08, -0.06]], [31.790, 22.877, 65.633, 45.908]),
        ("positive", [[0.08, 0.06]], [26.806, 17.391, 60.649, 40.422]),
    )
    def test_bounding_boxes_use_pixel_space_shear(self, shear, expected_box):
        # The cross terms must not scale with the image's aspect ratio.
        image_shape = _image_shape(65, 97)
        layer = layers.RandomShear(
            data_format=backend.config.image_data_format(),
            bounding_box_format="xyxy",
        )
        output = layer.transform_bounding_boxes(
            {
                "boxes": np.array([[[30.0, 21.0, 62.0, 42.0]]]),
                "labels": np.array([[1]]),
            },
            transformation={
                "shear_factor": np.array(shear),
                "input_shape": (1,) + image_shape,
            },
            training=True,
        )
        self.assertAllClose(output["boxes"], [[expected_box]], atol=1e-3)

    @parameterized.named_parameters(
        ("x_only", [[0.4, 0.0]]),
        ("y_only", [[0.0, 0.4]]),
        ("both_positive", [[0.3, 0.2]]),
        ("both_negative", [[-0.3, -0.2]]),
        ("opposite_signs", [[0.3, -0.2]]),
    )
    def test_bounding_boxes_match_sheared_image(self, shear):
        data_format = backend.config.image_data_format()
        x1, y1, x2, y2 = 70, 40, 110, 70
        image = np.zeros((1, 120, 200, 1), dtype="float32")
        image[0, y1:y2, x1:x2, 0] = 1.0
        if data_format == "channels_first":
            image = np.transpose(image, (0, 3, 1, 2))

        layer = layers.RandomShear(
            interpolation="nearest",
            fill_mode="constant",
            data_format=data_format,
            bounding_box_format="xyxy",
        )
        transformation = {
            "shear_factor": backend_utils.convert_tf_tensor(np.array(shear)),
            "input_shape": image.shape,
        }
        sheared_image = keras.ops.convert_to_numpy(
            layer.transform_images(image, transformation, training=True)
        )
        sheared_box = layer.transform_bounding_boxes(
            {
                "boxes": np.array([[[x1, y1, x2, y2]]], dtype="float32"),
                "labels": np.array([[1]]),
            },
            transformation=transformation,
            training=True,
        )["boxes"]

        rows, cols = np.nonzero(np.squeeze(sheared_image) > 0.5)
        # Each pixel covers `[i, i + 1]`, and nearest sampling can move the
        # edge of the rectangle by up to a pixel.
        self.assertAllClose(
            sheared_box,
            [[[cols.min(), rows.min(), cols.max() + 1, rows.max() + 1]]],
            atol=1.0,
        )

    @parameterized.named_parameters(
        ("rel_xyxy", "rel_xyxy", [0.25, 0.1, 0.625, 0.4]),
        ("xywh", "xywh", [2.0, 1.0, 3.0, 3.0]),
        ("center_xywh", "center_xywh", [3.5, 2.5, 3.0, 3.0]),
    )
    def test_bounding_boxes_keep_their_format(self, box_format, boxes):
        image_shape = _image_shape(10, 8)
        layer = layers.RandomShear(
            data_format=backend.config.image_data_format(),
            bounding_box_format=box_format,
        )
        transformation = {
            "shear_factor": np.array([[0.25, 0.0]]),
            "input_shape": (1,) + image_shape,
        }
        output = layer.transform_bounding_boxes(
            {"boxes": np.array([[boxes]]), "labels": np.array([[1]])},
            transformation=transformation,
            training=True,
        )
        expected = {
            "rel_xyxy": [1.125 / 8, 0.1, 4.875 / 8, 0.4],
            "xywh": [1.125, 1.0, 3.75, 3.0],
            "center_xywh": [3.0, 2.5, 3.75, 3.0],
        }[box_format]
        self.assertAllClose(output["boxes"], [[expected]])

    @parameterized.named_parameters(
        ("float32", "float32"),
        ("mixed_float16", "mixed_float16"),
    )
    def test_bounding_boxes_stay_finite_for_singular_shear(self, dtype):
        image_shape = _image_shape(10, 8)
        layer = layers.RandomShear(
            data_format=backend.config.image_data_format(),
            bounding_box_format="xyxy",
            dtype=dtype,
        )
        output = layer.transform_bounding_boxes(
            {
                "boxes": np.array(
                    [[[2.0, 1.0, 4.0, 3.0], [6.0, 4.0, 8.0, 6.0]]]
                ),
                "labels": np.array([[1, 2]]),
            },
            transformation={
                "shear_factor": np.array([[1.0, 1.0]]),
                "input_shape": (1,) + image_shape,
            },
            training=True,
        )
        boxes = keras.ops.convert_to_numpy(output["boxes"])
        self.assertTrue(np.all(np.isfinite(boxes)))
        self.assertTrue(np.all(boxes >= 0.0))
        self.assertTrue(np.all(boxes[..., ::2] <= 8.0))
        self.assertTrue(np.all(boxes[..., 1::2] <= 10.0))


def _image_shape(height, width):
    if backend.config.image_data_format() == "channels_last":
        return (height, width, 3)
    return (3, height, width)
