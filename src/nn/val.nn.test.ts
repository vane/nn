import { MathNN } from './math.nn';
import { ValNn } from './val.nn';

describe('nn/val.nn', () => {
  test('add 2 values', () => {
    const result = ValNn.new(2).add(ValNn.new(-2));
    expect(result.value).toEqual(0);
  });

  test('subtract 2 values', () => {
    const result = ValNn.new(2).sub(ValNn.new(-2));
    expect(result.value).toEqual(4);
  });

  test('multiply 2 values', () => {
    const result = ValNn.new(-2).mul(ValNn.new(-2));
    expect(result.value).toEqual(4);
  });

  test('divide 2 values', () => {
    const result = ValNn.new(-2).div(ValNn.new(-2));
    expect(result.value).toEqual(1);
  });

  test('exp value', () => {
    const result = ValNn.new(1).exp();
    expect(result.value).toEqual(Math.E);
  });

  test('tanh value', () => {
    const result = ValNn.new(0.101).tanh();
    expect(0.1 - result.value).toBeLessThan(0.0001);
  });

  test('pow value', () => {
    const result = ValNn.new(4).pow(4);
    expect(result.value).toEqual(256);
  });

  test('backward mul', () => {
    const result = ValNn.new(2).mul(ValNn.new(-3));
    result.backward();
    expect(result.value).toEqual(-6);
    expect(result.op?.right?.grad).toEqual(2);
    expect(result.op?.left?.grad).toEqual(-3);
  });

  test('backward div', () => {
    const result = ValNn.new(3).div(ValNn.new(-3));
    result.backward();
    expect(result.value).toEqual(-1);
    expect(result.op?.left.grad).toEqual(-0.3333333333333333);
    expect(result.op?.right?.grad).toEqual(-0.3333333333333333);
  });

  test('backward add / sub', () => {
    const result = ValNn.new(3).add(ValNn.new(1)).mul(ValNn.new(2));
    result.backward();
    expect(result.value).toEqual(8);
    expect(result.op?.left.grad).toEqual(2);
    expect(result.op?.right?.grad).toEqual(4);
    expect(result.op?.left.op?.left.grad).toEqual(2);
    expect(result.op?.left.op?.right?.grad).toEqual(2);
  });

  test('test backward', () => {
    const numOperations = 9;
    const x1 = ValNn.new(2);
    const x2 = ValNn.new(0);

    const w1 = ValNn.new(-3);
    const w2 = ValNn.new(1);

    const b = ValNn.new(6.88137);
    // 1->mul
    const x1w1 = x1.mul(w1);
    // 2->mul
    const x2w2 = x2.mul(w2);
    // 3->add
    const nn = x1w1.add(x2w2);
    // 4->add
    const n = nn.add(b);
    // 5->mul, 6->exp
    const e = ValNn.new(2).mul(n).exp();
    // 7->sub
    const i0 = e.sub(ValNn.new(1));
    // 8->add
    const i1 = e.add(ValNn.new(1));
    // 9->div
    const o = i0.div(i1);
    const backwardSize = o.backward();
    expect(MathNN.digit4(x1.grad)).toEqual(-1.5);
    expect(MathNN.digit4(x2.grad)).toEqual(0.5);
    expect(MathNN.digit4(w1.grad)).toEqual(1);
    expect(MathNN.digit4(w2.grad)).toEqual(0);
    expect(MathNN.digit4(i0.grad)).toEqual(0.1464);
    expect(MathNN.digit4(i1.grad)).toEqual(-0.1036);
    expect(MathNN.digit4(e.grad)).toEqual(0.0429);
    // check number of backward operations equal number of forward operations 9
    expect(backwardSize).toEqual(numOperations);
  });
});
