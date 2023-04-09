export class MathNN {
    static randomRange = (min: number, max: number): number => {
        if (min === max) return min;
        return Math.abs(Math.random() * (max - min) - max);
    }

    static digit2 = (value: number): number => {
        return Math.round(value * 100) / 100;
    }

    static digit3 = (value: number): number => {
        return Math.round(value * 1000) / 1000;
    }

    static digit4 = (value: number): number => {
        return Math.round(value * 10000) / 10000;
    }

    /**
     * Cumulative distribution function CDF
     * https://en.wikipedia.org/wiki/Cumulative_distribution_function
     */
    /**
     * Returns array with range list[0] - sum(list)
     * https://en.wikipedia.org/wiki/Càdlàg
     */
    static accumulate = (list: number[]): number[] => {
        const out = [list[0]]
        for (let i =1;i<list.length;i++) {
            out.push(list[i] + out[out.length - 1]);
        }
        return out;
    }

    static normalizeSum = (list: number[]): number[] => {
        const sum = list.reduce((a, b) => a+b, 0);
        return list.map(l => l/sum);
    }

    static softmax = (list: number[]): number[] => {
        const exp = list.map(a => Math.exp(a));
        const sum = exp.reduce((a, b) => a+b, 0);
        return exp.map((l) => l/sum);
    }

    static multinomialRepeat(list: number[], numSamples: number, randomGenerator: () => number): number[] {
        const a = this.accumulate(list);
        const out = [];
        for (let i =0;i<numSamples;i++) {
            out.push(this.bisect(a, randomGenerator()));
        }
        return out;
    }

    static multinomial(list: number[], numSamples: number, randomGenerator: () => number): number[] {
        if (list.length < numSamples) return [];
        // map of indexes
        const indexMap: {[key: number]: number} = {};
        for (let i =0;i<list.length;i++) {
            indexMap[list[i]] = i
        }

        // copy list
        const copy = list.slice();
        const out = [];
        let a, index;

        // accumulate values, pick index and remove value from copy
        for (let i =0;i<numSamples;i++) {
            a = this.accumulate(copy);
            index = this.bisect(a, randomGenerator());
            out.push(indexMap[copy[index]]);
            copy.splice(index, 1);
        }
        return out;
    }

    static smooth = (list: number[], value: number): number[] => {
        return list.map(v => v+value);
    }

    static log = (list: number[]): number[] => {
        return list.map(Math.log);
    }

    static logSumAbs = (list: number[]): number => {
        return -list.map(Math.log).reduce((a, b) => a+b, 0)
    }

    static logSumAbsNormalised = (list: number[]): number => {
        return this.logSumAbs(list)/list.length;
    }

    /**
     * Find value place in list
     */
    static bisect = (list: number[], value: number): number => {
        if (value < list[0]) return 0;
        if (value > list[list.length - 1]) return list.length - 1;
        let lo = 0;
        let hi = list.length - 1;
        let mid = -1;
        while (hi - lo > 1) {
            mid = Math.floor((hi + lo) / 2);
            if (value > list[mid]) {
                lo = mid;
            } else if (value < list[mid]) {
                hi = mid
            } else {
                return mid;
            }
        }
        if (value - list[lo] === 0) return lo;
        if (list[mid] - value < 0) return hi;
        return mid;
    }
}