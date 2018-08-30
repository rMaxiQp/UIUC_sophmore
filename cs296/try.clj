;;
;;Day 1
;;(def balance {"Zara" 3434.34, "Mahnaz" 123.22, "Daisy" 99.34, "Qadir" -19.08})
(ns x1)
(def x1 '(1,2,3,4))
(count x1)
(rest x1)
(cons 123 x1)
(second x1)
(cons 20 (cons 40 x1))
;;using namespace p2
(count x1)
(defn sum [xx]
  (if (empty? xx) 0 (+ (first xx) ( (rest xx)))))
(apply + x1)
(apply * x1)
(= x1 '(1 2 3 4))
(reduce + x1)
(def y1 x1)
(defn foo[x y] (+ x y))
(reduce foo '(1,2,3,4))
(defn bar[x y ] (+ x 10))
(reduce bar '(2 3))
(+ 1 2)
(+ 1 2 3)
(def v1 [1 ,2 ,3])
v1
(v1 2)
(map v1 '(1 2 2))
(filterv #(> % 5) v1)
(map inc v1)
(mapv inc v1)
(map inc v1)
(#(+ % 10) 285)
v1
;;
;;Day 2
;;
(hash "Howo")
(def h1{"emergency" 911})
(h1 "emergency")
(def h2(assoc h1 "Jenny" 87978))
(h2 "emergency")
(def coord{:island {:x 10 :y 30}})
((coord :island) :x)
(get-in [:island :y] coord)
(get-in coord [:island :y])
(update-in coord [:island :y] (fn [x] (* x 100)))
(keys coord)
(vals coord)
(def hh #{1 2 3 4 9})
(hh 3)
hh
(into #{} (list 11 23))
(into hh (list 12 33))
(defn maps [f s] (set (map f s)))
maps
(def h1 {:name "Gollum", :salary 34000, :title "coder"})
(def h2 (assoc h1 :hobby "Weiqi"))
h2
(defn bar [] {:x 10})
(merge h2 (bar))
(assoc h2 :my-key 1231)
(defn double [x] (* x 2))
(inc (double (inc 10)))
(-> 10 inc double inc)
(defn baz [] #(3 4 5))
(merge h2 (:y baz))
(require '[clojure.set :as set])
(set/intersection #{1 23 4} #{12 3 4})
(set/union #{23 1 21} #{12 33 41})
(sort (seq (set/union #{23 1 21} #{12 33 41})))
(ns-unmap 'x1 'h2)
;;
;;Day 3
;;
(def a 12.32)
(def x 10)
(def y x)
x
(def x [10 20 30])
(def y [1 2 3])
(def z x)
(def w [x y z])
(defrecord Pair [x y])
Pair
(def p1 (Pair. 10 20))
p1
(:x p1)
p1
(def p2 (Pair. 10 20))
p2
(= p1 p2)
(identical? p1 p2)
(def p3 p1)
(= p1 p3)
(identical? p1 p3)
(:x (:y (:y p1)))
(def p3 (Pair. 30 40))
(def p2 (Pair. 20 p3))
(def p1 (Pair. 10 p2))
(:x (:y (:y p1)))
(-> p1 :y :y :x)
(defrecord Triple [a b c])
(def x (Pair. 10 20))
(def y (Pair. x 40))
(def z (Triple. x (:x y) y))
z

;;
;; Day 4
;;
(ns hofs)

(defn dec[x] (- x 1))

(defn incList [x]
  (cond (empty? x) nil
        :fine-be-that-way (cons (inc (first x)) (incList (rest x)))))

(defn decList [x]
  (cond (empty? x) nil
        :keyword-doesnt-matter (cons (dec (first x)) (decList (rest x)))))

(map inc '(1 2 3))

(map inc '(1 2))

(incList '(1 2 3))

(defn mymap [f x]
  (cond (empty? x) nil
        true (cons (f (first x)) (mymap f (rest x)))))

(mymap dec '(1 2 3))

(defn foo [x] (if (> x 5) 3.14 12))

(foo 45)
(foo 2)

(defn yourmap [f & x]
  (cond (empty? (first x)) nil
        true (cons (apply f (map first x)) (yourmap f (map rest x)))))

(map first '((10 20 30) (20 4 7)))

(defn foo [a & x] (list a x))

(foo 10 20 30 40 50 60)

(Math/cos 1)
(Math/cos 100)

(defn cos [x](Math/cos x))

(defn fix[f x]
  (let [result(f x)]
    (cond (= result x) x
    :otherwise (fix f result))))

(fix cos 1)

(defn todoList [fs x]
  (cond (empty? fs) x
        true (todoList (rest fs) ((first fs) x))))

(todoList (list inc dec) 10)

;;
;; Day 5
;;

(defrecord ListZip [before after])

(defn current [z] (first (:after z)))
(defn forward [z]
  (ListZip. (cons (-> z :after first) (:before z)) (rest (:after z))))
(defn backward [x]
  (ListZip. (rest (:after x))
            (cons (-> x :before first) (:after x))))
(defn make-zipper [x] (ListZip. '() x))
(def z1 (make-zipper '(2 3 5 8)))
z1
(current z1)
z1
(current z1)
(forward z1)
z1
(current z1)
(forward (forward z1))
(def z2 (forward z1))
z2
(defn update [z f] (ListZip. (:before z) (cons (-> z :after first f) (-> z :after rest))))
(update z2 forward)
z2

;;
;; Day 6
;;

(ns stuff.bst)

(defrecord BST [left data right])

(defn bst "create a new BST."
  ([elt] (BST. nil elt nil))
  ([lc elt rc] (BST. lc elt rc)))

(defn find "Find a value in a BST, else return nil"
  [tree key]
  (cond
    (nil? tree) nil
    (= (:data tree) key) (:data tree)
    (< (:data tree) key) (find(:right tree) key)
    :otherwise           (find(:left tree) key)
    )
  )

(defn insert "insert a node at correct position"
  [tree key]
  (cond
    (nil? tree) (bst key)
    (> (:data tree) key) (bst (insert(:left tree) key) (:data tree) (:right tree))
    (< (:data tree) key) (bst (:left tree) (:data tree) (insert(:right tree) key))
    )
  )

;;
;; Day 7
;;

