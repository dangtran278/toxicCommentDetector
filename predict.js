const { spawn } = require('child_process');

const childPython = spawn('python', ['predict.py', 'You sussy baka! I will hit you with my big stick']);

childPython.stdout.on('data', (data) => {
    console.log(`${data}`);
})

// childPython.stderr.on('data', (data) => {
//     console.log(`stderr: ${data}`);
// })

// childPython.on('close', (code) => {
//     console.log(`child process exited on code: ${code}`);
// })