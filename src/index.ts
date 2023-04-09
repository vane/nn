import { MultiLayerNn } from './nn/multi.layer.nn';
import { ValNn } from './nn/val.nn';

class Foo {
  constructor() {
    console.log('foo');
    this.testBackward();
    this.testMultiLayer();
  }

  testBackward() {
    const x1 = ValNn.new(2);
    const x2 = ValNn.new(0);

    const w1 = ValNn.new(-3);
    const w2 = ValNn.new(1);

    const b = ValNn.new(6.8813735870195432);
    const x1w1 = x1.mul(w1);
    const x2w2 = x2.mul(w2);
    const nn = x1w1.add(x2w2);
    const n = nn.add(b);
    const ee = ValNn.new(2).mul(n);
    const e = ee.exp();
    const i0 = e.sub(ValNn.new(1));
    const i1 = e.add(ValNn.new(1));
    const o = i0.div(i1);
    o.backward();
    console.log(`o ${o}`);
    console.log(`i0 ${i0}`);
    console.log(`i1 ${i1}`);
    console.log(`e ${e}`);
    console.log(`ee ${ee}`);
    console.log(`n ${n}`);
    console.log(`nn ${nn}`);
    console.log(`x2w2 ${x2w2}`);
    console.log(`x1w1 ${x1w1}`);
    console.log(`w2 ${w2}`);
    console.log(`w1 ${w1}`);
    console.log(`x2 ${x2}`);
    console.log(`x1 ${x1}`);
  }

  testMultiLayer() {
    const ml = MultiLayerNn.new(3, [4, 4, 1]);
    const mlParams = ml.params;
    const xs = [
      [2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0]
    ];

    const ys = [1.0, -1.0, -1.0, 1.0];

    let ypred = [];
    let loss;
    for (let i = 0; i < 10; i++) {
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
      for (let j = 0; j < mlParams.length; j++) {
        mlParams[j].grad = 0;
      }
      loss.backward();
      for (let j = 0; j < mlParams.length; j++) {
        mlParams[j].value += -0.1 * mlParams[j].grad;
      }
      console.log(`${i} - ${loss.value}`);
    }
    console.log(`TARGET ${ys}`);
    console.log(`PRED ${mlParams.length} - ${ypred}`);
  }
}

new Foo();
