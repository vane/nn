import { MathNN } from './math.nn';

describe('nn/math.nn', () => {
  test('randomRange equal values', () => {
    const result = MathNN.randomRange(2, 2);
    expect(result).toBe(2);
  });

  test('randomRange', () => {
    const result = MathNN.randomRange(1, 2);
    expect(result).toBeGreaterThanOrEqual(1);
    expect(result).toBeLessThanOrEqual(2);
  });

  test('digit2', () => {
    const result = MathNN.digit2(0.12645);
    expect(result).toBe(0.13);
  });

  test('digit3', () => {
    const result = MathNN.digit3(0.12645);
    expect(result).toBe(0.126);
  });

  test('digit4', () => {
    const result = MathNN.digit4(0.12645);
    expect(result).toBe(0.1265);
  });

  test('accumulate', () => {
    const result = MathNN.accumulate([0, 1, 2, 3]);
    expect(result.reduce((a, b) => a + b, 0)).toBe(10);
  });

  test('softmax', () => {
    const result = MathNN.softmax([1, 2, 3, 4, 1, 2, 3]);
    const sum = result.reduce((a, b) => a + b, 0);
    expect(1 - sum).toBeLessThanOrEqual(0.000000000000001);
    expect(result.map(MathNN.digit3)).toStrictEqual([0.024, 0.064, 0.175, 0.475, 0.024, 0.064, 0.175]);
  });

  test('normalizeSum', () => {
    const result = MathNN.normalizeSum([1, 2, 3, 4, 5, 6, 7]);
    const sum = result.reduce((a, b) => a + b, 0);
    expect(1 - sum).toBeLessThanOrEqual(0.000000000000001);
  });

  test('bisect', () => {
    expect(MathNN.bisect([0.1, 0.9, 2, 5.2], 0.01)).toBe(0);
    expect(MathNN.bisect([0.1, 0.9, 2, 5.2], 0.1)).toBe(0);
    expect(MathNN.bisect([0.1, 0.9, 2, 5.2], 0.11)).toBe(1);
    expect(MathNN.bisect([0.1, 0.9, 2, 5.2], 0.91)).toBe(2);
    expect(MathNN.bisect([0.1, 0.9, 2, 5.2], 2)).toBe(2);
    expect(MathNN.bisect([0.1, 0.9, 2, 5.2], 2.1)).toBe(3);
    expect(MathNN.bisect([0.1, 0.9, 2, 5.2], 3)).toBe(3);
    expect(MathNN.bisect([0.1, 0.9, 2, 5.2], 5.21)).toBe(3);
  });

  test('bisect array', () => {
    const p = [0.1, 0.25, 0.4, 0.05, 0.2];
    const cdf = MathNN.accumulate(p);
    const result = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].map((r) => MathNN.bisect(cdf, r));
    expect(result).toStrictEqual([0, 1, 1, 2, 2, 2, 2, 3, 4, 4]);
  });
});
