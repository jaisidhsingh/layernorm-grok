# Toy math games for grokking

## 1. Sum and modulus: (A+B)%P

- $0 \leq A, B \leq P$
- $P= \text{const.}$

## 2. Sum and right shift: (A+B) >> P

- $0 \leq A, B \leq K$
- Since some $p >> q = \text{floor}(\frac{p}{2^q})$, we can write $(A+B) >> P = \text{floor}(\frac{A+B}{2^P})$
- $0 \leq A+ B \leq 2K \implies \frac{A+B}{2^P} \leq \frac{2K}{2^P}$
- We want the result to also be $K$ at max, so $\frac{2K}{2^P} \leq K + 1 \implies P \geq \log_2(\frac{2K}{K+1}) \implies P \geq 1$
- For convenience, we set $P = \text{ceil}(\log_2(\frac{2K}{K+1}))$

## 3. Sum and XOR: (A+B) ^ P (non-linear => hard?)

- $0 \leq A, B, P \leq \text{floor}(K/3)$, because
- $0 \leq (A+B) \oplus  P \leq A+B+P$

## 4. Sum and AND: (A+B) & P

- $0 \leq A, B, P \leq K \implies 0 \leq (A+B) \land P \leq K$
