export enum OpNn {
  ADD = '+',
  SUB = '-',
  MUL = '*',
  DIV = '/',
  TANH = 'tanh',
  POW = 'pow',
  EXP = 'exp'
}

export interface ValNnOp {
  left: ValNn;
  right?: ValNn;
  scalar?: number;
  op: OpNn;
}

export class ValNn {
  grad = 0;

  constructor(public value: number, readonly op?: ValNnOp) {}

  add = (val: ValNn) => {
    return new ValNn(this.value + val.value, {
      left: this,
      right: val,
      op: OpNn.ADD
    });
  };

  sub = (val: ValNn) => {
    return new ValNn(this.value - val.value, {
      left: this,
      right: val,
      op: OpNn.SUB
    });
  };

  mul = (val: ValNn) => {
    return new ValNn(this.value * val.value, {
      left: this,
      right: val,
      op: OpNn.MUL
    });
  };

  div = (val: ValNn) => {
    return new ValNn(this.value / val.value, {
      left: this,
      right: val,
      op: OpNn.DIV
    });
  };

  exp = () => {
    return new ValNn(Math.exp(this.value), {
      left: this,
      op: OpNn.EXP
    });
  };

  tanh = () => {
    return new ValNn(Math.tanh(this.value), {
      left: this,
      op: OpNn.TANH
    });
  };

  pow = (val: number) => {
    return new ValNn(this.value ** val, {
      left: this,
      scalar: val,
      op: OpNn.POW
    });
  };

  toString = () => {
    return `${this.value}`;
  };

  static new = (value: number): ValNn => {
    return new ValNn(value);
  };

  backward = (): number => {
    this.grad = 1;
    const ops: ValNn[] = [this];
    const visited = new Set();
    while (ops.length > 0) {
      const op = ops.shift();
      if (visited.has(op)) continue;
      if (op?.op) {
        visited.add(op);
        ValNn.backwardOne(op.op, op);
        op.op.right ? ops.push(op.op.left, op.op.right) : ops.push(op.op.left);
      }
    }
    return visited.size;
  };

  static backwardOne = (val: ValNnOp, next: ValNn): void => {
    switch (val.op) {
      case OpNn.ADD:
        val.left.grad += next.grad;
        val.right!.grad += next.grad;
        break;
      case OpNn.SUB:
        val.left.grad += next.grad;
        val.right!.grad += -1 * next.grad;
        break;
      case OpNn.DIV:
        val.left.grad += next.grad / val.right!.value;
        val.right!.grad += (-val.left.value / val.right!.value ** 2) * next.grad;
        break;
      case OpNn.MUL:
        val.left.grad += val.right!.value * next.grad;
        val.right!.grad += val.left.value * next.grad;
        break;
      case OpNn.POW:
        val.left.grad += val.scalar! * val.left.value ** (val.scalar! - 1) * next.grad;
        break;
      case OpNn.TANH:
        val.left.grad += (1 - next.value ** 2) * next.grad;
        break;
      case OpNn.EXP:
        val.left.grad += next.value * next.grad;
        break;
    }
  };
}
