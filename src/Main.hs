{-# LANGUAGE ScopedTypeVariables #-}
module Main where

import qualified Data.Map.Strict as Map
import Control.Monad.State.Strict
import System.Random

-- | Operations that make up the computational graph. `r` is the type
-- | of the result. `g` is the type of the gradient. These are usually
-- | both Double.
data Op r g = Add (Op r g) (Op r g) r g
            | Mult (Op r g) (Op r g) r g
            | Div (Op r g) (Op r g) r g
            | Exp (Op r g) r g
            | Neg (Op r g) r g
            | Log (Op r g) r g
            | Max (Op r g) (Op r g) r g
            | Var Int r g
            | Weight Int r g
            | Const Double g deriving (Show, Eq)

type Env = Map.Map Int Double

-- | Forward propogation, returning the graph decorated with the result
-- | after each step along with the final result.
forward :: Env -- Weight environment
        -> Env -- Var environment
        -> Op r g
        -> (Op Double g, Double)
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
forward wEnv vEnv (Neg x _ g) =
  let (op, res) = forward wEnv vEnv x
      res' = -res
  in (Neg op res' g, res')
forward wEnv vEnv (Exp x _ g) =
  let (op, res) = forward wEnv vEnv x
      res' = exp res
  in (Exp op res' g, res')
forward wEnv vEnv (Log x _ g) =
  let (op, res) = forward wEnv vEnv x
      res' = log res
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
backward :: Op Double g -> Double -> Op Double Double
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
backward (Neg x v _) grad =
  let gradx = -(val x) * grad
      op = backward x gradx
  in Neg op v grad
backward (Exp x v _) grad =
  let gradx = (exp (val x)) * grad
      op = backward x gradx
  in Exp op v grad
backward (Log x v _) grad =
  let gradx = (1/(val x)) * grad
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
relu op = max_ (const_ 0) op

-- A linear unit: a*x + b
linear :: Double -> Double -> Double -> Op () ()
linear a x b = add (mult (const_ a) (const_ x)) (const_ b)

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
unit inputs act = act (foldr f (weight 0) inputs)
  where f cur acc = add (mult (weight 0) cur) acc

-- | e^op / (1 + e^op)
sigmoid :: Op () () -> Op () ()
sigmoid op = div_ (const_ 1.0) (add (const_ 1.0) (exp_ (neg op)))

-- | Squared loss function. (expected - actual)^2
squaredLoss :: Op () () -- Expected value
            -> Op () () -- Actual value
            -> Op () ()
squaredLoss expected actual =
  let diff = add (actual) (neg expected)
  in mult diff diff

crossEntropy :: Op () () -> Op () () -> Op () ()
crossEntropy expected actual =
  neg (mult expected (log_ actual))
  
div_, mult, add, max_ :: Op () () -> Op () () -> Op () ()
div_ l r = Div l r () ()
mult l r = Mult l r () ()
add l r = Add l r () ()
max_ l r = Max l r () ()

const_ :: Double -> Op () ()
const_ v = Const v ()

exp_, log_ :: Op () () -> Op () ()
exp_ x = Exp x () ()
log_ x = Log x () ()

neg :: Op () () -> Op () ()
neg x = Neg x () ()

var, weight :: Int -> Op () ()
var n = Var n () ()
weight n = Weight n () ()

val :: Op Double g-> Double
val (Log _ v _) = v
val (Mult _ _ v _ ) = v
val (Add _ _ v _ ) = v
val (Var _ v _) = v
val (Weight _ v _) = v
val (Max _ _ v _) = v
val (Const v _) = v
val (Div _ _ v _) = v
val (Exp _ v _) = v
val (Neg _ v _) = v

xor :: Op () ()
xor =
  let hidden = layer [var 1, var 2] relu 2
  in unit hidden relu

-- | Walk the graph, giving fresh names to weights and returning
-- | a map of weights names to weights.
initWeights :: Op r g -> State (Int, Env, StdGen) (Op r g)
initWeights (Weight _ v g) = do
  (i, env, rand) <- get
  let (w, rand') = randomR (-0.5, 0.5) rand
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
initWeights (Log x r g) = do
  op <- initWeights x
  return (Log op r g)
initWeights (Neg x r g) = do
  op <- initWeights x
  return (Neg op r g)
initWeights (Max l r v g) = do
  op1 <- initWeights l
  op2 <- initWeights r
  return (Max op1 op2 v g)
initWeights x = return x

collectGradients :: Op r Double -> State Env (Op r Double)
collectGradients (Weight i v g) = do
  env <- get
  -- If a single weight unit outputs to multiple nodes,
  -- add the gradients together.
  put (Map.insertWith (+) i g env)
  return (Weight i v g)
collectGradients (Div l r v g) = do
  op1 <- collectGradients l
  op2 <- collectGradients r
  return (Div op1 op2 v g)
collectGradients (Add l r v g) = do
  op1 <- collectGradients l
  op2 <- collectGradients r
  return (Add op1 op2 v g)
collectGradients (Mult l r v g) = do
  op1 <- collectGradients l
  op2 <- collectGradients r
  return (Mult op1 op2 v g)
collectGradients (Exp x r g) = do
  op <- collectGradients x
  return (Exp op r g)
collectGradients (Log x r g) = do
  op <- collectGradients x
  return (Log op r g)
collectGradients (Neg x r g) = do
  op <- collectGradients x
  return (Neg op r g)
collectGradients (Max l r v g) = do
  op1 <- collectGradients l
  op2 <- collectGradients r
  return (Max op1 op2 v g)
collectGradients x = return x

gradDescent :: Env -- One training data
            -> Op r g -- Neural net
            -> Double -- Learning rate
            -> State Env Double -- Learned weights
gradDescent training net lRate = do
  w <- get
  let (f, loss) = forward w training net
  let b = backward f 1
  let grad = execState (collectGradients b) Map.empty
  let updates = Map.map (* (-lRate)) grad
  put (Map.unionWith (+) w updates)
  return loss

gd :: [Env] -> Op r g -> Double -> Int -> State Env [Double]
gd training net lRate n = do
  losses <- forM training (\t -> gradDescent t net lRate)
  let avgLoss = avg losses
  if n > 0
    then fmap (avgLoss :) (gd training net lRate (n-1))
    else return [avgLoss]

avg :: [Double] -> Double
avg fs = (sum fs) / fromIntegral ((length fs))

initOp :: Op () () -> StdGen -> (Op () (), Env)
initOp op rand =
  let (initNet, (_, wEnv, _)) = runState (initWeights op) (0, Map.empty, rand)
  in (initNet, wEnv)

main :: IO ()
main = do

  rand <- getStdGen
  let (xorInit, (_, wEnv, _)) = runState (initWeights xor) (0, Map.empty, rand)
  print wEnv
  let withLoss = squaredLoss (var 3) xorInit
  --let withLoss = crossEntropy xorInit (var 3)
  let training = cycle [ Map.fromList [(1, 1), (2, 1), (3, 0.0)]
                       , Map.fromList [(1, 0), (2, 0), (3, 0.0)]
                       , Map.fromList [(1, 1), (2, 0), (3, 1.0)]
                       , Map.fromList [(1, 0), (2, 1), (3, 1.0)]
                       ]
  print $ fmap (\d -> let (_, res) = forward wEnv d xorInit in res) (take 4 training)
  {-let knownGood =
        Map.fromList [ (0,1.0530994)
                     , (1,0.9493089)
                     , (2,-0.94930893)
                     , (3,2.0358142e-8)
                     , (4,1.0728915)
                     , (5,-0.93179905)
                     , (6,0.931799)
                     , (7,2.6292934e-8)
                     , (8,1.4203037e-4)
                     ]
  -}
  let (r, weights) = runState (gd (take 80 training) withLoss 0.0001 5000) wEnv
  putStrLn "Losses"
  print r
  putStrLn "Weights"
  print weights
  putStrLn "Graph"
  print xorInit
  print $ fmap (\d -> let (_, res) = forward weights d xorInit in res) (take 4 training)
