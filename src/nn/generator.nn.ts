import { RandomSeed, create } from 'random-seed';

export class GeneratorNn {
  private generator: RandomSeed;

  constructor(seed?: number) {
    this.generator = create(seed ? seed.toString() : undefined);
  }

  random = (): number => {
    return this.generator.random();
  };

  randomList = (amount: number): number[] => {
    const out = [];
    for (let i = 0; i < amount; i++) out.push(this.random());
    return out;
  };

  randomRange = (min: number, max: number) => {
    return this.generator.floatBetween(min, max);
  };
}
