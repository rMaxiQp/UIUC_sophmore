(ns bsts.core-test
  (:use [midje.sweet]
        [bsts.core])
  (:import [bsts.core BST]))

(facts "About add."
       (fact "It adds leaf nodes."
             (add nil 10) => (BST. nil 10 nil))
       (fact "It doesn't duplicate stuff."
             (add (add nil 10) 10) => (BST. nil 10 nil)) )

(facts "About find."
       (fact "It finds things.  Duh."
             (bst-find (reduce add nil '(40 20 50)) 40) => true
             (bst-find (reduce add nil '(40 20 50)) 20) => true
             (bst-find (reduce add nil '(40 20 50)) 50) => true
             (bst-find (reduce add nil '(40 20 50)) 45) => false))


(deftest a-test
  (testing "FIXME, I fail."
    (is (= 0 1))))
