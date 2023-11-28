
"use strict";

let IsInRemoteControl = require('./IsInRemoteControl.js')
let Load = require('./Load.js')
let Popup = require('./Popup.js')
let RawRequest = require('./RawRequest.js')
let AddToLog = require('./AddToLog.js')
let GetProgramState = require('./GetProgramState.js')
let IsProgramSaved = require('./IsProgramSaved.js')
let GetRobotMode = require('./GetRobotMode.js')
let IsProgramRunning = require('./IsProgramRunning.js')
let GetSafetyMode = require('./GetSafetyMode.js')
let GetLoadedProgram = require('./GetLoadedProgram.js')

module.exports = {
  IsInRemoteControl: IsInRemoteControl,
  Load: Load,
  Popup: Popup,
  RawRequest: RawRequest,
  AddToLog: AddToLog,
  GetProgramState: GetProgramState,
  IsProgramSaved: IsProgramSaved,
  GetRobotMode: GetRobotMode,
  IsProgramRunning: IsProgramRunning,
  GetSafetyMode: GetSafetyMode,
  GetLoadedProgram: GetLoadedProgram,
};
