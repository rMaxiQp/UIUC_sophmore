#include "Chara.h"

string Chara::getStatus() const{
  if(q.empty()) return "Empty";
  if(q.size() < 4) return "Light";
  if(q.size() < 7) return "Moderate";
  return "Heavy";
}

void Chara::addToQueue(string name){
  q.push(name);
}
string Chara::popFromQueue(){
  string temp = q.front();
  q.pop();
  return temp;
}
