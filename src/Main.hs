module Main where

-- | Operations that make up the computational graph. `r` is the type
-- | of the result. `g` is the type of the gradient. These are usually
-- | both Float.
data Op r g = Add (Op r g) (Op r g) r g
            | Mult (Op r g) (Op r g) r g
            | Max (Op r g) (Op r g) r g
            | Const Float g deriving (Show, Eq)

-- | Forward propogation, returning the graph decorated with the result
-- | after each step along with the final result.
forward :: Op r g -> (Op Float g, Float)
forward (Add l r v g) =
  let (op1, res1) = forward l
      (op2, res2) = forward r
      res' = res1 + res2
  in
    (Add op1 op2 res' g, res')
forward (Mult l r v g) =
  let (op1, res1) = forward l
      (op2, res2) = forward r
      res' = res1 * res2
  in
    (Mult op1 op2 res' g, res')
forward (Max l r v g) =
  let (op1, res1) = forward l
      (op2, res2) = forward r
      res' = max res1 res2
  in
    (Max op1 op2 res' g, res')
forward (Const v g ) = (Const v g, v)

-- | Backprop. Takes a graph that has been forward propogated and
-- | a starting gradient (usually 1.0) and fills in the gradients
-- | with respect to the final output.
backward :: Op Float g -> Float -> Op Float Float
backward (Add l r v _) grad =
  let op1 = backward l grad
      op2 = backward r grad
  in
    Add op1 op2 v grad
backward (Mult l r v _) grad =
  let gradl = (val r) * grad
      gradr = (val l) * grad
      op1 = backward l gradl
      op2 = backward r gradr
  in Mult op1 op2 v grad
backward (Max l r v g) grad =
  let lval = val l
      rval = val r
      (lgrad, rgrad) = if lval > rval then (grad, 0) else (0, grad)
      op1 = backward l lgrad
      op2 = backward r rgrad
  in Max op1 op2 v grad
backward (Const v _) grad =
  Const v grad

-- A ReLu (rectified linear) unit: max {0, op}
relu :: Op () () -> Op () ()
relu op = Max (Const 0 ()) op () ()

-- A linear unit: a*x + b
linear :: Float -> Float -> Float -> Op () ()
linear a x b = Add (Mult (Const a ()) (Const x ()) () ()) (Const b ()) () ()

val :: Op Float g-> Float
val (Mult _ _ v _ ) = v
val (Add _ _ v _ ) = v
val (Const v _) = v

main :: IO ()
main = do
  let g = relu (linear (-3) 2 3)
  let (gForward, res) = forward g
  print g
  print gForward
  print (backward gForward 1)
