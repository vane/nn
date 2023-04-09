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
});
