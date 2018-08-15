const tf = require("@tensorflow/tfjs");
require("@tensorflow/tfjs-node");

test("should be able to train a model", () => {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  model.compile({ loss: "meanSquaredError", optimizer: "sgd" });

  const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
  const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

  return model.fit(xs, ys, { epochs: 10 }).then(() => {
    model.predict(tf.tensor2d([5], [1, 1])).print();
  });
});
