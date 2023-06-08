import { MathNN } from './math.nn';
import { ValNn } from './val.nn';

export class NeuronNn {
  weights: ValNn[];
  b: ValNn;
  constructor(numInputs: number) {
    this.weights = [];
    for (let i = 0; i < numInputs; i++) {
      this.weights.push(ValNn.new(MathNN.randomRange(-1, 1)));
    }
    this.b = ValNn.new(MathNN.randomRange(-1, 1));
  }

  forward(value: ValNn[]) {
    let sum = ValNn.new(0);
    for (let i = 0; i < this.weights.length; i++) {
      sum = sum.add(this.weights[i].mul(value[i]));
    }
    sum = sum.add(this.b);
    return sum.tanh();
  }

  zeroGrad() {
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i].grad = 0;
    }
    this.b.grad = 0;
  }

  addGrad(value: number) {
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i].value += value * this.weights[i].grad;
    }
    this.b.value = value * this.b.grad;
  }

  static new = (numInputs: number): NeuronNn => {
    return new NeuronNn(numInputs);
  };
}
