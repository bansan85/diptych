function loadPyModule(module, callback) {
  xmlHttp = new XMLHttpRequest();
  xmlHttp.open( 'GET', '/' + module + '.py', true );
  xmlHttp.overrideMimeType("text/plain; charset=x-user-defined"); 

  xmlHttp.onload = function() {
    document.source = this.responseText
    document.module = module
    pyodide.runPython(`
import js

import sys
sys.path.insert(0, '/')

print ("python" + js.document.module + ".py")

with open(js.document.module + ".py", "w") as fd:
  fd.write(js.document.source)
`);

    pyodide.runPython(`
import ` + module + `
`);
    callback();
  }

  xmlHttp.send('');
}

function loadPyModules(modules, callbackStart, callbackEnd, finalCallback, i = 0) {
  if (Object.is(modules.length - 1, i)) {
    callbackStart(modules[i])
    loadPyModule(modules[i], finalCallback)
  } else {
    callbackStart(modules[i])
    loadPyModule(modules[i], () => {
      callbackEnd(modules[i])
      loadPyModules(modules, callbackStart, callbackEnd, finalCallback, i + 1);
    })
  }
}
