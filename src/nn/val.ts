export enum Op {
    ADD = '+',
    SUB = '-',
    MUL = '*',
    DIV = '/',
    TANH = 'tanh',
    POW = 'pow',
    EXP = 'exp'
}

export interface ValOp {
    left: Val;
    right?: Val;
    scalar?: number;
    op: Op;
}

export class Val {
    value: number;
    grad = 0;

    constructor(value: number, readonly op?: ValOp) {
        this.value = value;
    }

    add(val: Val) {
        return new Val(this.value + val.value, {
            left: this,
            right: val,
            op: Op.ADD
        });
    }

    sub(val: Val) {
        return new Val(this.value - val.value, {
            left: this,
            right: val,
            op: Op.SUB
        });
    }

    mul(val: Val) {
        return new Val(this.value * val.value, {
            left: this,
            right: val,
            op: Op.MUL
        });
    }

    div(val: Val) {
        return new Val(this.value / val.value, {
            left: this,
            right: val,
            op: Op.DIV
        });
    }

    exp() {
        return new Val(Math.exp(this.value), {
            left: this,
            op: Op.EXP
        })
    }

    tanh() {
        return new Val(Math.tanh(this.value), {
            left: this,
            op: Op.TANH
        })
    }

    pow(val: number) {
        return new Val(Math.pow(this.value, val), {
            left: this,
            scalar: val,
            op: Op.POW
        });
    }

    toString = () => {
        return `Val(value=${this.value}, grad=${this.grad})`
    }

    printOps = (): string => {
        let ops: Val[] = [this];
        let out: string[] = [];
        while (ops.length > 0) {
            const op = ops.shift();
            if (op?.op) {
                if (op.op.right) {
                    out.push(`${op.op.left.value} ${op.op.op} ${op.op.right.value} = ${op.value}
left(grad=${op.op.left.grad})
right(grad=${op.op.right.grad})
result(grad=${op.grad})
`)
                    ops.push(op.op.left, op.op.right);
                } else {
                    out.push(`${op.op.op}(${op.op.left.value}) = ${op.value}
left(grad=${op.op.left.grad})
result(grad=${op.grad})
`)
                    ops.push(op.op.left);
                }
            }
        }
        return out.reverse().map((v, i) => `${i + 1} -> ${v}`).join('\n');
    }

    static new = (value: number): Val => {
        return new Val(value);
    }

    backward(): void {
        this.grad = 1;
        let ops: Val[] = [this];
        const visited = new Set();
        while (ops.length > 0) {
            const op = ops.shift();
            if (visited.has(op)) continue;
            if (op?.op) {
                visited.add(op);
                Val.backwardOne(op.op, op);
                if (op.op.right) {
                    ops.push(op.op.left, op.op.right);
                } else {
                    ops.push(op.op.left);
                }
            }
        }
    }

    static backwardOne = (val: ValOp, next: Val): void => {
        switch (val.op) {
            case Op.ADD:
            case Op.SUB:
                val.left.grad += next.grad;
                val.right!.grad += next.grad;
                break;
            case Op.DIV:
                val.left.grad += 1 / val.right!.value  * next.grad;
                val.right!.grad += -1 / (2 * val.left.value) * next.grad;
                break;
            case Op.MUL:
                val.left.grad += (val.right!.value * next.grad);
                val.right!.grad += val.left.value * next.grad;
                break;
            case Op.POW:
                val.left.grad += val.scalar! * Math.pow(val.left.value, val.scalar! - 1) * next.grad
                break;
            case Op.TANH:
                val.left.grad += (1 - Math.pow(next.value, 2)) * next.grad;
                break;
            case Op.EXP:
                val.left.grad += next.value * next.grad;
                break;
        }
    }
}