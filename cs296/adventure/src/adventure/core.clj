(ns adventure.core
  (:require [clojure.core.match :refer [match]]
            [clojure.string :as str])
  (:gen-class))

(def the-map
  {:foyer{:desc "You just escaped from zombies. Holding a There's nothing in the room except a table, a television and a remote."
           :title "at foyer"
           :dir {:left :kitchen :right :bedroom :out :outside}
           :contents #{:remote}}
   :kitchen {:desc "Look like a kitchen. Uh...It smells terrible. Is that gas leak?"
              :title "in the kitchen"
              :dir {:right :foyer :downstair :basement}
              :contents #{:knife}}
   :basement{:desc "It's dark down here."
              :title "The basement is dark. I cannot see anything."
              :dir {:upstair :kitchen :left :garage :on :light-basement}
              :contents #{:screwdriver :crow-bar}}
   :bedroom {:desc "There's a bed. Take a nap?"
              :title "in the bedroom"
              :dir {:left :foyer :upstair :attic :right :closet}
              :contents #{:pillow :driver-license}}
   :attic {:desc "There is a lot of stuff here."
              :title "in the attic"
              :dir {:downstair :bedroom}
              :contents #{:chest :car-key}}
   :garage {:desc "Nice! A car."
              :title "in the garage"
              :dir {:right :basement}
              :contents #{:car}}
   :outside {:desc "The door behind you is locked. Your brain is about to eaten...See you in next life..."
              :title "surrounded by zombies..."
              :dir {}
              :contents #{:zombies}}
   :closet {:desc "Hum...Weird. There's a book and a key."
              :title "in the closet"
              :dir {:left :bedroom}
              :contents #{:book :key}}
   })

(def obj
  {:car {:desc "maybe you can use it to escape from the hord..."}
   :book {:desc "a book with red cover...nevermind, it's blood"}
   :chest {:desc "a locked chest with a tag 'The Hitchhiker's Guide to the Galaxy'."}
   :car-key {:desc "a key to success!"}
   :driver-license {:desc "at least you have a license now..."}
   :screwdriver {:desc "No more screw please..."}
   :pillow {:desc "about to take a nap?"}
   :knife {:desc "a sharp knife. Fair item to end your life..."}
   :remote {:desc "hmmm...a TV remote. Maybe you can find something useful inside of it."}
   :crow-bar {:desc "Ha! the best weapon ever."}
   :key {:desc "a suspicious key that has no use."}
   })

(def adventurer
  {:location :foyer
   :inventory :nothing
   :seen #{}
   :torch 0})

(defn complain [player]
  (do
    (println "********************************************************************************")
    (println "TADA! Here is the non-sense")
    (println "This is just a little game made with clojure...")
    (println "After having a lot of fun with syntax issues (a lot)")
    (println "We finally develope something that is... at least playable")
    (println "This final project is fun, and there are still a lot to implement.\n Have something to work on during the break now...")
    (println "HOHOHO~ Merry Christmas")
    (println "********************************************************************************")
    player))

(defn credit []
  (println "_____________________________________________________________________________")
  (println "credit to zq2, junliwu2, yuyangl2...")
  (println "thank you for your time")
  (println "_____________________________________________________________________________"))

(defn quit []
  (credit)
  (System/exit 0))

(defn status [player]
  (let [location (player :location) item (player :inventory)]
    (print (str "You are " (-> the-map location :title) ". "))
    (when-not ((player :seen) location)
      (println (-> the-map location :desc)))
    (when-not ((player :seen) location)
      (println (str "You find ") (-> the-map location :contents)))
    (println (str "You are holding"item))
    (if (= location :outside)
      (do (quit) player))
    (update-in player [:seen] #(conj % location))))

(defn to-keywords [commands]
  (mapv keyword (str/split commands #"[.,?! ]+")))

(defn go [dir player]
  (let [location (player :location)
        dest (->> the-map location :dir dir)]
    (cond (nil? dest) (do (println "You can't go that way.") player)
          (= dest :outside) (assoc-in player [:location] dest)
          (= (-> player :torch) 0) (do (println "it's too dark here, you should find some batteries for your torch first") player)
          :else (assoc-in player [:location] dest))))

(defn drive [player]
  (if (= (player :location) :garage)
    (do ((println "Successfully driving of the zombie zone.")
         (quit)) player)))

(defn grab [item player]
    (println (str "Grabbing" item))
    (assoc-in player [:inventory] item))

(defn light [player]
  (println "Get two batteries...You put them into your torch and you can explore the house now")
  (assoc-in player [:torch] 1))

(defn use_item [player]
  (let [item (player :inventory) light (player :torch)]
    (if (= item :nothing) (do (println "You are holding nothing...Don't lie to yourself.") player)
    (cond
      (= item :book) (do (println "Zombies move slower than cars. And what is the answer to life, the universe and everything?") player)
      (= item :chest) (do (println "ITS LOCKED, YOU NEED TO open IT.") player)
      (= item :remote) (do (println "Get two batteries...You put them into your torch and you can explore the house now.") (def newPlayer (update-in player [:torch] inc))newPlayer)
      (= item :knife) (do (println "You use the knife commit suicide.") (quit))
      (= item :pillow) (do (println "You take a nap and gas leak terminates your life.") (quit))
      (= item :car-key) (drive player)
      :else (do (println (str "Nah, it doesn't work.")))))))

(defn puzzle [player]
  (println "There are 10 types of people in this world, those who understand binary and those who dont...")
  (println "What is the answer?")
  (let [answer "101010" your-answer (read-line)]
    (if (= answer your-answer) (do ((println "you open the chest and earn the eternity...")
                                    (quit))))
      (do ((println "you fail the test but the chest opens anyway...chest becomes a monster and eats you...") (quit)))
    ))

(defn open [player]
  (let [item (player :inventory)]
    (if (= item :chest) (puzzle player))))

(defn discard [player]
  (do (def newPlayer(assoc-in player [:inventory] :nothing))newPlayer))

(defn help [player]
  (do
    (print "+")
    (print "------------------------------------------------------------------------")
    (println "+")
    (println "TYPE up, down, left, right TO NAVIGATE")
    (println "TYPE look TO SHOW INVENTORY AND LOCATION")
    (println "TYPE grab + ITEM NAME TO PICK, ONLY ONE AT A TIME")
    (println "TYPE discard + ITEM NAME TO DROP")
    (println "TYPE examine + ITEM NAME TO CHECK ITEM DESCRIPTION")
    (println "TYPE use TO USE (IF APPLICABLE) THE ITEM YOU ARE HOLDING")
    (println "TYPE dir TO SHOW POSSIBLE WAYS THAT YOU CAN GO TO")
    (println "TYPE slay TO BE A WARRIOR OR FOOL")
    (println "TYPE quit IF YOU ARE BORED")
    (println "TYPE ONE CHAR COMMAND TO SEE SOME NON-SENSE")
    (print "+")
    (print "------------------------------------------------------------------------")
    (println "+")
  ) player
)

(defn slay [player]
  (if (= (-> player :inventory) :crow-bar)
    (do (println "With the power of crow-bar, you leave the door and defeat the horde. People honor the crow-bar as the hero (Yes. not you... whoops).") (quit) player)
    (do
      (println "You break out the door...")
      (def newPlayer (assoc-in player [:location] :outside)) newPlayer)))


(defn examine [item player]
  (do (println (str (-> obj item :desc))) player))

(defn show [player]
  (let [location (player :location)]
    do(println (-> the-map location :dir))player))

(defn respond [player command]
  (match command
         ;;helping functions
         [:look] (update-in player [:seen] #(disj % (-> player :location)))
         [:help] (help player)
         [:dir] (show player)

         ;;direction functions
         [:left] (go :left player)
         [:right] (go :right player)
         [:up] (go :upstair player)
         [:down] (go :downstair player)
         [:out] (go :out player)

         ;;use item
         [:use] (use_item player)
         [:open] (open player)

         ;;hold item
         [:grab :screwdriver] (grab :screwdriver player)
         [:grab :car-key] (grab :car-key player)
         [:grab :key] (grab :key player)
         [:grab :book] (grab :book player)
         [:grab :chest] (grab :chest player)
         [:grab :remote] (grab :remote player)
         [:grab :crow-bar] (grab :crow-bar player)
         [:grab :knife] (grab :knife player)
         [:grab :pillow] (grab :pillow player)

         ;;check item
         [:examine :screwdriver] (examine :screwdriver player)
         [:examine :car-key] (examine :car-key player)
         [:examine :key] (examine :key player)
         [:examine :book] (examine :book player)
         [:examine :chest] (examine :chest player)
         [:examine :remote] (examine :remote player)
         [:examine :crow-bar] (examine :crow-bar player)
         [:examine :knife] (examine :knife player)
         [:examine :pillow] (examine :pillow player)

         [:slay] (slay player)
         [:quit] (quit)

         [:c] (complain player)

         ;;delete whatever you are holding
         [:discard] (discard player)

         _ (do (println "I don't understand you.")
               player)

         ))

(defn -main
  [& args]
  (println "Treat this as walking dead's new game's demo")
  (println "the final version would be released...pretty soon")
  (println "_________________________________________________________________________\n")
  (loop [local-map the-map
         local-player adventurer]
    (let [pl (status local-player)
          _  (println "What do you want to do?")
          command (read-line)]
      (recur local-map (respond pl (to-keywords command))))))
