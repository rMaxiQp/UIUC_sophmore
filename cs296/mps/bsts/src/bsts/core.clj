(ns bsts.core)

(defrecord BST [left data right])

(defn bst "Create a new BST."
  ([elt] (BST. nil elt nil))
  ([lc elt rc] (BST. lc elt rc)))

