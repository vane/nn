import { LayerNn } from './layer.nn';
import { ValNn } from './val.nn';

export class MultiLayerNn {
  layers: LayerNn[];
  constructor(numInputs: number, numOutputs: number[]) {
    this.layers = [];
    let inputs = numInputs;
    for (let i = 0; i < numOutputs.length; i++) {
      this.layers.push(LayerNn.new(inputs, numOutputs[i]));
      inputs = numOutputs[i];
    }
  }

  forward(value: ValNn[]): ValNn[] {
    for (let i = 0; i < this.layers.length; i++) {
      value = this.layers[i].forward(value);
    }
    return value;
  }

  get params(): ValNn[] {
    const out = [];
    for (let i = 0; i < this.layers.length; i++) {
      out.push(...this.layers[i].params);
    }
    return out;
  }

  static new = (numInputs: number, numOutputs: number[]): MultiLayerNn => {
    return new MultiLayerNn(numInputs, numOutputs);
  };
}
