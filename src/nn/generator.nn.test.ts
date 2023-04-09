import { GeneratorNn } from './generator.nn';
import { MathNN } from './math.nn';

describe('nn/generator.nn', () => {
  test('pseudo random with seed', () => {
    const g = new GeneratorNn(213769);
    expect(g.random()).toBe(0.45641034373106826);
  });

  test('pseudo random 3 numbers with seed', () => {
    const g = new GeneratorNn(213769);
    expect(g.randomList(3)).toStrictEqual([0.45641034373106826, 0.2913891340193575, 0.8878157347445593]);
  });

  test('multinomialRepeat with pseudo random', () => {
    const g = new GeneratorNn(213769);
    const norm = MathNN.normalizeSum([1, 2, 3, 4, 5, 6, 7]);
    const result = MathNN.multinomialRepeat(norm, 3, g.random);
    expect(result).toStrictEqual([4, 3, 6]);
  });

  test('multinomial without repeat with pseudo random', () => {
    const g = new GeneratorNn(213769);
    const norm = MathNN.normalizeSum([1, 2, 3, 4, 5, 6, 7]);
    const result = MathNN.multinomial(norm, 7, g.random);
    expect(result).toStrictEqual([4, 3, 6, 5, 2, 1, 0]);
  });
});
