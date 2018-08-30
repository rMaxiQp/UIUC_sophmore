
vector<string> NoProblem(int start, vector<int> created, vector<int> needed) {
  vector<string> v;
  if(start >= needed[0]){
    v.push_back("No problem! :D");
  }
  else
  {
    v.push_back("No problem. :(");
  }
    for(size_t t = 1; t < needed.size(); t++)
    {
      if(created[t] >= needed[t]){
        v.push_back("No problem! :D");
        if(created[t] < needed[t]){
          start = start + (created[t] - needed[t]);
        }
      }
      else
      {
        v.push_back("No problem. :(");
        cout<<"created: "<< created[t] << " needed: "<< needed[t] << " start: "<< start <<'\n';
      }
    }
    return v;
}
