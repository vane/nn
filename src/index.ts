import { ValNn } from './nn/valNn';

class Foo {
  constructor() {
    console.log('foo');
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
}

new Foo();
