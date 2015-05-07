(ns neural-network.core
  (:refer-clojure :exclude  [* - + == /])
  (:require  [clojure.core.matrix :refer :all]
             [clojure.core.matrix.operators :refer :all]))

(def input-neurons [1 0])
(def input-hidden-strengths [[0.12 0.20 0.13]
                             [0.01 0.02 0.03]])
(def hidden-neurons [0 0 0])
(def hidden-output-strengths [ [0.15 0.16]
                               [0.02 0.03]
                               [0.01 0.02]])
(def targets [0 1])


(defn activation-fn [x]
  (Math/tanh x))
(defn dactivation-fn [y]
  (- 1.0 (* y y)))
(defn output-deltas [targets outputs]
  (* (mapv dactivation-fn outputs)
     (- targets outputs)))
(defn layer-activation [inputs strengths]
  (mapv activation-fn
        (mapv #(reduce + %)
              (* inputs (transpose strengths)))))

(def new-hidden-neurons
  (layer-activation input-neurons input-hidden-strengths))
(def new-output-neurons
  (layer-activation new-hidden-neurons hidden-output-strengths))

(defn -main []
  (println (layer-activation input-neurons input-hidden-strengths))
  (println (output-deltas targets new-output-neurons)))
