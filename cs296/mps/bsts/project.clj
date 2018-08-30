(defproject bsts "0.1.0-SNAPSHOT"
  :description "A simple binary search tree"
  :url ""
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :profiles {:dev {:dependencies [[midje "1.8.3" :exclusions [org.clojure/clojure]]]}}
  :plugins [[lein-midje "3.2.1"]]
  :dependencies [[org.clojure/clojure "1.8.0"]])
