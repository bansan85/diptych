function loadPyModule(module) {
  return new Promise((resolve, reject) => {
    const xmlHttp = new XMLHttpRequest();
    xmlHttp.open('GET', `/diptych/${module}.py`, true);
    xmlHttp.overrideMimeType('text/plain; charset=x-user-defined');

    xmlHttp.onload = () => {
      if (xmlHttp.status === 200) {
        document.source = xmlHttp.responseText;
        document.module = module;
        pyodide.runPython(`
import js

import os
import sys

sys.path.insert(0, '/')

print("python" + js.document.module + ".py")

if not os.path.exists("diptych"):
  os.makedirs("diptych")
with open("diptych/" + js.document.module + ".py", "w") as fd:
  fd.write(js.document.source)
        `);

        pyodide.runPython(`
import diptych.${module}
        `);
        resolve();
      } else {
        reject();
      }
    };

    xmlHttp.send('');
  });
}

function loadPyModules(modules, callbackStart, callbackEnd, i = 0) {
  if (Object.is(modules.length - 1, i)) {
    callbackStart(modules[i]);
    return loadPyModule(modules[i]);
  }

  callbackStart(modules[i]);
  return loadPyModule(modules[i]).then(() => {
    callbackEnd();
    return loadPyModules(modules, callbackStart, callbackEnd, i + 1);
  });
}
