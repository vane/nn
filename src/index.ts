import { MultiLayerNn } from './nn/multi.layer.nn';
import { ValNn } from './nn/val.nn';

class Foo {
  constructor() {
    this.testMultiLayer();
  }

  testMultiLayer() {
    const ml = MultiLayerNn.new(3, [4, 4, 1]);
    const xs = [
      [2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0]
    ];

    const ys = [1.0, -1.0, -1.0, 1.0];

    let ypred, loss;
    for (let i = 0; i < 100; i++) {
      // forward
      ypred = [];
      for (let j = 0; j < xs.length; j++) {
        const v = ml.forward(xs[j].map(ValNn.new));
        ypred.push(v[0]);
      }
      // loss
      loss = ValNn.new(0);
      for (let j = 0; j < ys.length; j++) {
        loss = loss.add(ValNn.new(ys[j]).sub(ypred[j]).pow(2));
      }
      // backward
      ml.zeroGrad();
      loss.backward();
      ml.addGrad(-0.1);
      //console.log(`${i} - ${loss.value}`);
    }
    console.log(`TARGET ${ys}`);
    console.log(`PRED ${ypred}`);
  }
}

new Foo();
