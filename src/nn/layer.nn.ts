import { NeuronNn } from './neuron.nn';
import { ValNn } from './val.nn';

export class LayerNn {
  neurons: NeuronNn[];
  constructor(numInputs: number, numOutputs: number) {
    this.neurons = [];
    for (let i = 0; i < numOutputs; i++) {
      this.neurons.push(NeuronNn.new(numInputs));
    }
  }

  forward(value: ValNn[]): ValNn[] {
    const out = [];
    for (let i = 0; i < this.neurons.length; i++) {
      out.push(this.neurons[i].forward(value));
    }
    return out;
  }

  zeroGrad() {
    for (let i = 0; i < this.neurons.length; i++) {
      this.neurons[i].zeroGrad();
    }
  }

  addGrad(value: number) {
    for (let i = 0; i < this.neurons.length; i++) {
      this.neurons[i].addGrad(value);
    }
  }

  static new = (numInputs: number, numOutputs: number): LayerNn => {
    return new LayerNn(numInputs, numOutputs);
  };
}
