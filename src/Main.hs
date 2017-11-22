{-# LANGUAGE ScopedTypeVariables #-}
module Main where

import qualified Data.Map.Strict as Map
import Control.Monad.State.Strict
import System.Random

-- | Operations that make up the computational graph. `r` is the type
-- | of the result. `g` is the type of the gradient. These are usually
-- | both Float.
data Op r g = Add (Op r g) (Op r g) r g
            | Mult (Op r g) (Op r g) r g
            | Div (Op r g) (Op r g) r g
            | Exp (Op r g) r g
            | Max (Op r g) (Op r g) r g
            | Var Int r g
            | Weight Int r g
            | Const Float g deriving (Show, Eq)

type Env = Map.Map Int Float

-- | Forward propogation, returning the graph decorated with the result
-- | after each step along with the final result.
forward :: Env -- Weight environment
        -> Env -- Var environment
        -> Op r g
        -> (Op Float g, Float)
forward wEnv vEnv (Add l r _ g) =
  let (op1, res1) = forward wEnv vEnv l
      (op2, res2) = forward wEnv vEnv r
      res' = res1 + res2
  in
    (Add op1 op2 res' g, res')
forward wEnv vEnv (Mult l r _ g) =
  let (op1, res1) = forward wEnv vEnv l
      (op2, res2) = forward wEnv vEnv r
      res' = res1 * res2
  in
    (Mult op1 op2 res' g, res')
forward wEnv vEnv (Exp x _ g) =
  let (op, res) = forward wEnv vEnv x
      res' = exp res
  in (Exp op res' g, res')
forward wEnv vEnv (Div l r _ g) =
  let (op1, res1) = forward wEnv vEnv l
      (op2, res2) = forward wEnv vEnv r
      res' = res1 / res2
  in (Div op1 op2 res' g, res')
forward wEnv vEnv (Max l r _ g) =
  let (op1, res1) = forward wEnv vEnv l
      (op2, res2) = forward wEnv vEnv r
      res' = max res1 res2
  in
    (Max op1 op2 res' g, res')
forward _ vEnv (Var name _ g) =
  let Just v = Map.lookup name vEnv -- Blow up if the name isn't bound.
  in (Var name v g, v)
forward wEnv _ (Weight name _ g) =
  let Just v = Map.lookup name wEnv -- Blow up if the name isn't bound.
  in (Weight name v g, v)
forward _ _ (Const v g ) = (Const v g, v)

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
backward (Div l r v _) grad =
  let gradl = grad / (val r)
      gradr = ((-val l) * grad) / (val r * val r)
      op1 = backward l gradl
      op2 = backward r gradr
  in Div op1 op2 v grad
backward (Exp x v _) grad =
  let gradx = exp (val x)
      op = backward x gradx
  in Exp op v grad
backward (Max l r v _) grad =
  let lval = val l
      rval = val r
      (lgrad, rgrad) = if lval > rval then (grad, 0) else (0, grad)
      op1 = backward l lgrad
      op2 = backward r rgrad
  in Max op1 op2 v grad
backward (Var name r _) grad =
  Var name r grad
backward (Weight name r _) grad =
  Weight name r grad
backward (Const v _) grad =
  Const v grad

-- A ReLu (rectified linear) unit: max {0, op}
relu :: Op () () -> Op () ()
relu op = Max (Const 0 ()) op () ()

-- A linear unit: a*x + b
linear :: Float -> Float -> Float -> Op () ()
linear a x b = Add (Mult (Const a ()) (Const x ()) () ()) (Const b ()) () ()

-- | Create a fully connected layer.
layer :: [Op () ()] -- Inputs to the layer
      -> Activation -- Activation function for each unit
      -> Int        -- Number of units in the layer
      -> [Op () ()]
layer inputs act units =
  replicate units (unit inputs act)

type Activation = Op () () -> Op () ()

unit :: [Op () ()] -- Inputs to the unit
     -> Activation -- Activation function
     -> Op () ()
unit inputs act = act (foldr f (weight 0 ) inputs)
  where f cur acc = add (mult (weight 0) cur) acc

-- | e^op / (1 + e^op)
sigmoid op = div_ (exp_ op) (add (const_ 1) (exp_ op))

div_ l r = Div l r () ()
mult l r = Mult l r () ()
add l r = Add l r () ()
var n = Var n () ()
weight n = Weight n () ()
max_ l r = Max l r () ()
const_ v = Const v ()
exp_ x = Exp x () ()

val :: Op Float g-> Float
val (Mult _ _ v _ ) = v
val (Add _ _ v _ ) = v
val (Var _ v _) = v
val (Weight _ v _) = v
val (Max _ _ v _) = v
val (Const v _) = v
val (Div _ _ v _) = v
val (Exp _ v _) = v

xor :: Op () ()
xor =
  let hidden = layer [var 1, var 2] relu 2
  in unit hidden sigmoid

-- | Walk the graph, giving fresh names to weights and returning
-- | a map of weights names to weights.
initWeights :: Op r g -> State (Int, Env, StdGen) (Op r g)
initWeights (Weight _ v g) = do
  (i, env, rand) <- get
  let (w, rand') = randomR ((-1.0), 1.0) rand
  put (i+1, Map.insert i w env, rand')
  return (Weight i v g)
initWeights (Div l r v g) = do
  op1 <- initWeights l
  op2 <- initWeights r
  return (Div op1 op2 v g)
initWeights (Add l r v g) = do
  op1 <- initWeights l
  op2 <- initWeights r
  return (Add op1 op2 v g)
initWeights (Mult l r v g) = do
  op1 <- initWeights l
  op2 <- initWeights r
  return (Mult op1 op2 v g)
initWeights (Exp x r g) = do
  op <- initWeights x
  return (Exp op r g)
initWeights (Max l r v g) = do
  op1 <- initWeights l
  op2 <- initWeights r
  return (Max op1 op2 v g)
initWeights x = return x

  
main :: IO ()
main = do
  rand <- getStdGen
  let (xorInit, (_, wEnv, _)) = runState (initWeights xor) (0, Map.empty, rand)
  print xorInit
  let vEnv = Map.fromList [(1, 1), (2, 1)]
  let (xorForward, res) = forward wEnv vEnv xorInit
  putStrLn $ "Res: " ++ show res
  putStrLn "Forward"
  print xorForward
  putStrLn "Backward"
  print (backward xorForward 1)
