var mu = (function () {
  'use strict';

  function array_equals(arr,arr2){
    // Joe on Stack Overflow 27/12/2014.
    // In
    // array Array mandatory
    // Out Boolean
    // Returns true if two arrays are equal, and false otherwise.
    // http://stackoverflow.com/questions/7837456/comparing-two-arrays-in-javascript

    // If the other array is a falsy value, return.
    if (!arr2)
    return false;

    // Compare lengths.
    if (arr.length != arr2.length)
    return false;

    for (let i = 0, l=arr.length; i < l; i++){
      // Check if we have nested arr2s.
      if (arr[i] instanceof Array && arr2[i] instanceof Array){
        // Recurse into the nested arr2s.
        if (!array_equals(arr[i],arr2[i]))
        return false;
      }
      else if (arr[i] != arr2[i]){
        // Warning - two different object instances will never be equal:
        // {x:20} != {x:20}.
        return false;
      }
    }
    return true;
  }

  function index_item_1st_occurs(arr,a){
    // Tom Collins 1/2/2015.
    // In
    // a Array, Boolean, Number, or String mandatory
    // Out Integer
    // Returns the index at which the given argument a firsts occur. It is more
    // robust than indexOf functionality because it will match arguments
    // consisting of arrays, strings, and booleans as well as numbers. It will
    // not match arbitrary objects, however (see second example in testing).

    var typeofa = typeof a;
    var instanceofarraya = a instanceof Array;
    var idx = -1;
    var i = 0;
    while (i < arr.length)
    {
      if (typeof arr[i] == typeofa){
        if(instanceofarraya && arr[i] instanceof Array){
          if (array_equals(arr[i],a)){
            idx = i;
            i = arr.length - 1;
          }
        }
        else {
          if (arr[i] == a){
            idx = i;
            i = arr.length - 1;
          }
        }
      }
      i=i+1;
    }
    return idx;
  }

  function index_item_1st_doesnt_occur(arr,a){
    // Tom Collins 1/2/2015.
    // In
    // a Array, Boolean, Number, or String mandatory
    // Out Integer
    // Returns the index at which the given argument a first does not occur. It
    // is robust in the sense that it will match arguments consisting of arrays,
    // strings, and booleans as well as numbers. It will not match arbitrary
    // objects, however (see second example in testing).

    var typeofa = typeof a;
    var instanceofarraya = a instanceof Array;
    var idx = -1;
    var i = 0;
    while (i < arr.length)
    {
      if (!(typeof arr[i] == typeofa) ||
            (instanceofarraya && !(arr[i] instanceof Array))){
        idx = i;
        i = arr.length - 1;
      }
      else {
        if(instanceofarraya && arr[i] instanceof Array){
          if (!array_equals(arr[i],a)){
            idx = i;
            i = arr.length - 1;
          }
        }
        else {
          if (!(arr[i] == a)){
            idx = i;
            i = arr.length - 1;
          }
        }
      }
      i=i+1;
    }
    return idx;
  }

  Array.prototype.equals = function(a){
    return array_equals(this,a)
  };

  Array.prototype.index_item_1st_occurs = function(a){
    return index_item_1st_occurs(this,a)
  };

  Array.prototype.index_item_1st_doesnt_occur = function(a){
    return index_item_1st_doesnt_occur(this,a)
  };

  // Utility.
  function append_ontimes_to_time_signatures(time_sigs_array, crotchets_per_bar){
    // Tom Collins 26/2/2015.
    // This function appends ontimes to rows of the time-signature table. Added
    // an optional argument crotchets_per_bar, so that in the event of an
    // anacrusis, the first bar is assigned the correct ontime.

    if (crotchets_per_bar == undefined){
      var ontime = 0;
    }
    else {
      var ontime = -crotchets_per_bar;
    }
    time_sigs_array[0]["ontime"] = ontime;
    var i = 1;
    var n = time_sigs_array.length;
    while (i < n) {
      var c = (time_sigs_array[i]["barNo"] - time_sigs_array[i - 1]["barNo"])*time_sigs_array[i - 1]["topNo"]*
      4/time_sigs_array[i - 1]["bottomNo"];
      var d = time_sigs_array[i - 1]["ontime"] + c;
      time_sigs_array[i]["ontime"] = d;
      i=i+1;
    }
    return time_sigs_array;
  }

  function row_of_max_ontime_leq_ontime_arg(ontime, time_sigs_array){
    // Tom Collins 17/10/2014.
    // In
    // ontime Number mandatory
    // time_sigs_array Array mandatory
    // Out Array
    // This function returns the row (in a list of time signatures) of the
    // maximal ontime less than or equal to the ontime argument.

    var ontime_out = time_sigs_array[0];
    var i = 0;
    var n = time_sigs_array.length;
    while (i < n) {
      if (ontime < time_sigs_array[i]["ontime"]){
        ontime_out = time_sigs_array[i - 1];
        i = n - 1;
      }
      else if (ontime == time_sigs_array[0]["ontime"]){
        ontime_out = time_sigs_array[i];
        i = n - 1;
      }
      else if (i == n - 1){
        ontime_out = time_sigs_array[i];
      }
      i=i+1;
    }
    return ontime_out;
  }

  /**
   * Given an ontime and a time-signature array (with ontimes included), this
   * function returns the bar number and beat number of that ontime. Bar numbers
   * are one-indexed, meaning the bar number of an ontime in an anacrusis is zero.
   *
   * @author Tom Collins
   * @comment 17th October 2014
   * @param {number} ontime - An ontime (time counting in crotchet beats starting
   * from 0 for bar 1 beat 1) for which we want to know the corresponding bar and
   * beat number.
   * @param {TimeSignature[]} time_sigs_array - An array of time signatures.
   * @return {number[]} An array whose first element is a bar number and whose
   * second element is a beat number.
   *
   * @example
   *     var tsArr = [
   *   {
   *     "barNo": 1,
   *     "topNo": 4,
   *     "bottomNo": 4,
   *     "ontime": 0
   *   },
   *   {
   *     "barNo": 3,
   *     "topNo": 3,
   *     "bottomNo": 4,
   *     "ontime": 8
   *   }
   * ];
   * bar_and_beat_number_of_ontime(11, tsArr);
   * â†’
   * [4, 1]
   */
  function bar_and_beat_number_of_ontime(ontime, time_sigs_array){
    var n = time_sigs_array.length;
    var relevant_row = row_of_max_ontime_leq_ontime_arg(ontime, time_sigs_array);
    if (ontime >= 0) {
      var excess = ontime - relevant_row["ontime"];
      var local_beat_bar = relevant_row["topNo"]*4/relevant_row["bottomNo"];
      var a = [
        relevant_row["barNo"] + Math.floor(excess/local_beat_bar),
        (excess % local_beat_bar) + 1
      ];
    }
    else {
      var anacrusis_beat = time_sigs_array[0]["topNo"] + ontime + 1;
      var a = [0, anacrusis_beat];
    }
    return a;
  }

  const lookup = [
    { "sign": "G", "line": 2, "name": "treble clef"},
    { "sign": "G", "line": 2, "clefOctaveChange": 1, "name": "treble clef 8va" },
    { "sign": "G", "line": 2, "clefOctaveChange": 2, "name": "treble clef 15ma" },
    { "sign": "G", "line": 2, "clefOctaveChange": -1, "name": "treble clef 8vb" },
    { "sign": "G", "line": 1, "name": "French violin clef" },
    { "sign": "C", "line": 1, "name": "soprano clef" },
    { "sign": "C", "line": 2, "name": "mezzo-soprano clef" },
    { "sign": "C", "line": 3, "name": "alto clef" },
    { "sign": "C", "line": 4, "name": "tenor clef" },
    { "sign": "C", "line": 4, "name": "baritone clef (C clef)" },
    { "sign": "F", "line": 4, "name": "bass clef" },
    { "sign": "F", "line": 4, "clefOctaveChange": 1, "name": "bass clef 8va" },
    { "sign": "F", "line": 4, "clefOctaveChange": 2, "name": "bass clef 15ma" },
    { "sign": "F", "line": 4, "clefOctaveChange": -1, "name": "bass clef 8vb" },
    { "sign": "F", "line": 4, "clefOctaveChange": -2, "name": "bass clef 15mb" },
    { "sign": "F", "line": 3, "name": "baritone clef (F clef)" },
    { "sign": "F", "line": 5, "name": "subbass clef 15mb" },
    // These last two do not seem to be supported.
    { "sign": "percussion", "line": 2, "name": "percussion clef" },
    { "sign": "TAB", "line": 0, "name": "tablature" }
  ];

  function clef_sign_and_line2clef_name(sign, line, clef_octave_change){

    var i = 0;
  	while (i < lookup.length){
  		if (lookup[i].sign == sign &&
  				lookup[i].line == line &&
          (clef_octave_change == undefined ||
           lookup[i].clefOctaveChange &&
           lookup[i].clefOctaveChange == clef_octave_change)){
  			var clef_name = lookup[i].name;
        i = lookup.length - 1;
  		}
      i++;
  	}
  	if (clef_name == undefined){
  		return "unknown";
  	}
  	else {
  		return clef_name;
  	}
  }

  function convert_1st_bar2anacrusis_val(bar_1, divisions){
    // Tom Collins 25/2/2015.
    // In
    // bar_1 Object mandatory
    // divisions Integer mandatory
    // Out Array
    // This function works out how long an anacrusis contained in bar_1 should
    // last.

    // Get top and bottom number from time signature, to work out how long a full
    // first bar should last.
    if (bar_1.attributes){
      var attributes = bar_1.attributes;
      for (let j = 0; j < attributes.length; j++){
        if (attributes[j].time){
          // Assuming there is only one time per attribute...
          var time_sig_1 = {};
          time_sig_1.topNo = parseInt(attributes[j].time[0].beats[0]);
          time_sig_1.bottomNo = parseInt(attributes[j].time[0]['beat-type'][0]);
          }
      }
    }
    if (time_sig_1 == undefined) {
      console.log('It was not possible to find a time signature in the first ' +
                  'bar of the top staff.');
      console.log('Returning default values for the anacrusis and crotchets '+
                  'bar, which may be wrong.');
      return [0, 4];
    }

    var anacrusis = 0;
    var crotchets_per_bar = 4*time_sig_1.topNo/time_sig_1.bottomNo;
    var dur_in_1st_bar_should_be = divisions*crotchets_per_bar;
    var ontime = 0;

    // Get backup value.
    if (bar_1.backup){
      var backups = bar_1.backup;
      }
    else {
      backups = [];
    }

    // Increment over the notes.
    if (bar_1.note){
      var notes = bar_1.note;
      for (let note_index = 0; note_index < notes.length; note_index++){
        if (notes[note_index].grace == undefined){
          // This is the integer duration expressed in MusicXML.
          var duration = parseInt(notes[note_index].duration[0]);
          var offtime = ontime + duration;
          // Correct rounding errors in the offtime values.
          // If the note is a second, third, etc. note of a chord, then do
          // not increment the ontime variable.
          if (note_index < notes.length - 1 && notes[note_index + 1].chord);
          else { // Do increment the ontime value.
            ontime = offtime;
          }
        }
      }
    }
    var compar = ontime/(backups.length + 1);
    if (compar != dur_in_1st_bar_should_be){
      anacrusis = -compar/divisions;
    }
    return [anacrusis, crotchets_per_bar];
  }

  function default_page_and_system_breaks(staff_and_clef_names, final_bar_no){
    // Tom Collins 1/3/2015.
    // In
    // staff_and_clef_names Array mandatory
    // final_bar_no Integer mandatory
    // Out Array
    // If the page_breaks and system_breaks variables are empty, this function
    // will populate them with default values based on the number of staves and
    // bars.

    var page_breaks = [];
    var system_breaks = [];
    var nos_staves = staff_and_clef_names.length;
    switch (nos_staves){
      case 1:
        var sbreak = 4;
        var pbreak = 10*sbreak;
        break;
      case 2:
        var sbreak = 4;
        var pbreak = 5*sbreak;
        break;
      case 3:
        var sbreak = 4;
        var pbreak = 3*sbreak;
        break;
      case 4:
        var sbreak = 4;
        var pbreak = 2*sbreak;
        break;
      case 5:
        var sbreak = 4;
        var pbreak = 2*sbreak;
        break;
      case 6:
        var sbreak = 4;
        var pbreak = 2*sbreak;
        break;
      default:
        var sbreak = 4;
        var pbreak = sbreak;
        break;
    }
    var curr_bar = sbreak;
    while (curr_bar < final_bar_no){
      if (curr_bar%pbreak == 0){
        page_breaks.push(curr_bar + 1);
      }
      else {
        system_breaks.push(curr_bar + 1);
      }
      curr_bar = curr_bar + sbreak;
    }
    return [page_breaks, system_breaks];
  }

  function group_grace_by_contiguous_id(grace_array){
    // Tom Collins 18/2/2015.
    // In
    // grace_array Array mandatory
  	// An array of grace notes is the input to this function. The function groups
    // these grace notes into new arrays whose membership is determined by
    // contiguity of the id fields. This is to make sure that if several grace
    // notes precede an ordinary note, these are grouped together and (later)
    // attached to this ordinary note.

    var ga = grace_array.sort(sort_points_asc_by_id);
    if (ga.length > 0){
      var gag = [[ga[0]]];
      var gj = 0;
      for (let gi = 1; gi < ga.length; gi++){
        if (parseFloat(ga[gi].ID) ==
            parseFloat(gag[gj][gag[gj].length - 1].ID) + 1){
          gag[gj].push(ga[gi]);
        }
        else {
          gag.push([ga[gi]]);
          gj++;
        }
      }
    }
    else {
      var gag = [];
    }
    return gag;
  }

  const midi_residue_lookup_array = [
    [0, 0], [1, 0], [2, 1], [3, 2],
    [4, 2], [5, 3], [6, 3], [7, 4],
    [8, 4], [9, 5], [10, 6], [11, 6]
  ];

  function guess_morphetic_in_c_major(mnn){
    // Tom Collins 15/10/2014.
    // In
    // mnn Integer mandatory
    // Out Integer
    // This function takes a MIDI note number as its only argument. It
    // attempts to guess the corresponding morphetic pitch number, assuming
    // a key of or close to C major.

    var octave = Math.floor(mnn/12 - 1);
    var midi_residue = mnn - 12*(octave + 1);

    var midi_residue_idx = 0;
    var n = midi_residue_lookup_array.length;
    var i = 0;
    while (i < n){
      if (midi_residue == midi_residue_lookup_array[i][0]){
        midi_residue_idx = i;
        i = n - 1;
      }
      i=i+1;
    }
    var mpn_residue = midi_residue_lookup_array[midi_residue_idx][1];
    var a = mpn_residue + 7*octave + 32;
    return a;
  }

  const fifth_steps_lookup_array = [
    // Major keys.
    [[-6, 0], 6, 4], [[-5, 0], -1, -1],
    [[-4, 0], 4, 2], [[-3, 0], -3, -2],
    [[-2, 0], 2, 1], [[-1, 0], -5, -3],
    [[0, 0], 0, 0], [[1, 0], 5, 3],
    [[2, 0], -2, -1], [[3, 0], 3, 2],
    [[4, 0], -4, -2], [[5, 0], 1, 1],
    [[6, 0], -6, -3],
    // Minor keys.
    [[-3, 5], 6, 4], [[-2, 5], -1, -1],
    [[-1, 5], 4, 2], [[0, 5], -3, -2],
    [[1, 5], 2, 1], [[2, 5], -5, -3],
    [[3, 5], 0, 0], [[4, 5], 5, 3],
    [[5, 5], -2, -1], [[6, 5], 3, 2],
    [[-6, 5], 3, 2], [[7, 5], -4, -2],
    [[-5, 5], -4, -2], [[8, 5], 1, 1],
    [[-4, 5], 1, 1], [[9, 5], -6, -3]
  ];
  function guess_morphetic(mnn, fifth_steps, mode){
    // Tom Collins 15/10/2014.
    // In
    // mnn Integer mandatory
    // fifth_steps Integer mandatory
    // mode Integer mandatory
    // This function takes a MIDI note number and a key (represented by steps on
    // the circle of fiths, and mode). It attempts to guess the corresponding
    // morphetic pitch number, given the key.

    var fifth_steps_idx = 0;
    var n = fifth_steps_lookup_array.length;
    var i = 0;
    while (i < n){
      if (fifth_steps == fifth_steps_lookup_array[i][0][0] &&
          mode == fifth_steps_lookup_array[i][0][1]){
        fifth_steps_idx = i;
        i = n - 1;
      }
      i=i+1;
    }
    var trans = fifth_steps_lookup_array[fifth_steps_idx].slice(1);
    var z = mnn + trans[0];
    var w = guess_morphetic_in_c_major(z);
    var a = w - trans[1];
    return a;
  }

  const pitch_class_lookup_array = [
    [[12, 6], "B#"], [[0, 0], "C"], [[0, 1], "Dbb"],
    [[13, 6], "B##"], [[1, 0], "C#"], [[1, 1], "Db"],
    [[2, 0], "C##"], [[2, 1], "D"], [[2, 2], "Ebb" ],
    [[3, 1], "D#"], [[3, 2], "Eb"], [[3, 3], "Fbb"],
    [[4, 1], "D##"], [[4, 2], "E"], [[4, 3], "Fb"],
    [[5, 2], "E#"], [[5, 3], "F"], [[5, 4], "Gbb"],
    [[6, 2], "E##"], [[6, 3], "F#"], [[6, 4], "Gb"],
    [[7, 3], "F##"], [[7, 4], "G"], [[7, 5], "Abb"],
    [[8, 4], "G#"], [[8, 5], "Ab"],
    [[9, 4], "G##"], [[9, 5], "A"], [[9, 6], "Bbb"],
    [[-2, 0], "Cbb"], [[10, 5], "A#"], [[10, 6], "Bb"],
    [[11, 5], "A##"], [[11, 6], "B"], [[-1, 0], "Cb"]
  ];

  function midi_note_morphetic_pair2pitch_and_octave(mnn, mpn){
    // Tom Collins 15/10/2014.
    // In
    // mnn Integer mandatory
    // mpn Integer mandatory
    // Out String
    // This function converts a pair consisting of a MIDI note number and a
    // morphetic pitch number into a string consisting of a note's pitch and
    // octave.

    var octave = Math.floor((mpn - 32)/7);
    var midi_residue = mnn - 12*(octave + 1);
    var mpn_residue = mpn - (7*octave + 32);

    var pitch_class_idx = undefined;
    var n = pitch_class_lookup_array.length;
    var i = 0;
    while (i < n){
      if (midi_residue == pitch_class_lookup_array[i][0][0] &&
          mpn_residue == pitch_class_lookup_array[i][0][1]){
        pitch_class_idx = i;
        i = n - 1;
      }
      i=i+1;
    }
    var a = pitch_class_lookup_array[pitch_class_idx][1] + octave;
    return a;
  }

  const lookup$1 = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
  ];

  function mnn2pitch_simple(MNN){
    // Tom Collins 6/1/2016.
    // In
    // metadata Integer mandatory
    // Out String
    // This function converts a MIDI note number into a pitch class and octave.
    // It does so in a completely naive manner (no consideration of global or
    // local key), but this is handy for things like Tone.js playback, which tend
    // to prefer "C" to "B#", "C#" to "Db" (I think), and "G" to "F##".


    var octave = Math.floor(MNN/12 - 1);
    var MNN_mod_12 = MNN % 12;
    return lookup$1[MNN_mod_12] + octave //.toString();
  }

  var keysLookup = [
    { "nosSymbols": 0, "mode": "major", "keyName": "C major" },
    { "nosSymbols": 1, "mode": "major", "keyName": "G major" },
    { "nosSymbols": 2, "mode": "major", "keyName": "D major" },
    { "nosSymbols": 3, "mode": "major", "keyName": "A major" },
    { "nosSymbols": 4, "mode": "major", "keyName": "E major" },
    { "nosSymbols": 5, "mode": "major", "keyName": "B major" },
    { "nosSymbols": 6, "mode": "major", "keyName": "F# major" },
    { "nosSymbols": 7, "mode": "major", "keyName": "C# major" },
    { "nosSymbols": 8, "mode": "major", "keyName": "G# major" },
    { "nosSymbols": 9, "mode": "major", "keyName": "D# major" },
    { "nosSymbols": 10, "mode": "major", "keyName": "A# major" },
    { "nosSymbols": 11, "mode": "major", "keyName": "E# major" },
    { "nosSymbols": -1, "mode": "major", "keyName": "F major" },
    { "nosSymbols": -2, "mode": "major", "keyName": "Bb major" },
    { "nosSymbols": -3, "mode": "major", "keyName": "Eb major" },
    { "nosSymbols": -4, "mode": "major", "keyName": "Ab major" },
    { "nosSymbols": -5, "mode": "major", "keyName": "Db major" },
    { "nosSymbols": -6, "mode": "major", "keyName": "Gb major" },
    { "nosSymbols": -7, "mode": "major", "keyName": "Cb major" },
    { "nosSymbols": -8, "mode": "major", "keyName": "Fb major" },
    { "nosSymbols": -9, "mode": "major", "keyName": "Bbb major" },
    { "nosSymbols": -10, "mode": "major", "keyName": "Ebb major" },
    { "nosSymbols": -11, "mode": "major", "keyName": "Abb major" },
    { "nosSymbols": 0, "mode": "minor", "keyName": "A minor" },
    { "nosSymbols": 1, "mode": "minor", "keyName": "E minor" },
    { "nosSymbols": 2, "mode": "minor", "keyName": "B minor" },
    { "nosSymbols": 3, "mode": "minor", "keyName": "F# minor" },
    { "nosSymbols": 4, "mode": "minor", "keyName": "C# minor" },
    { "nosSymbols": 5, "mode": "minor", "keyName": "G# minor" },
    { "nosSymbols": 6, "mode": "minor", "keyName": "D# minor" },
    { "nosSymbols": 7, "mode": "minor", "keyName": "A# minor" },
    { "nosSymbols": 8, "mode": "minor", "keyName": "E# minor" },
    { "nosSymbols": 9, "mode": "minor", "keyName": "B# minor" },
    { "nosSymbols": 10, "mode": "minor", "keyName": "F## minor" },
    { "nosSymbols": 11, "mode": "minor", "keyName": "C## minor" },
    { "nosSymbols": -1, "mode": "minor", "keyName": "D minor" },
    { "nosSymbols": -2, "mode": "minor", "keyName": "G minor" },
    { "nosSymbols": -3, "mode": "minor", "keyName": "C minor" },
    { "nosSymbols": -4, "mode": "minor", "keyName": "F minor" },
    { "nosSymbols": -5, "mode": "minor", "keyName": "Bb minor" },
    { "nosSymbols": -6, "mode": "minor", "keyName": "Eb minor" },
    { "nosSymbols": -7, "mode": "minor", "keyName": "Ab minor" },
    { "nosSymbols": -8, "mode": "minor", "keyName": "Db minor" },
    { "nosSymbols": -9, "mode": "minor", "keyName": "Gb minor" },
    { "nosSymbols": -10, "mode": "minor", "keyName": "Cb minor" },
    { "nosSymbols": -11, "mode": "minor", "keyName": "Fb minor" },
    { "nosSymbols": 0, "mode": "ionian", "keyName": "C ionian" },
    { "nosSymbols": 1, "mode": "ionian", "keyName": "G ionian" },
    { "nosSymbols": 2, "mode": "ionian", "keyName": "D ionian" },
    { "nosSymbols": 3, "mode": "ionian", "keyName": "A ionian" },
    { "nosSymbols": 4, "mode": "ionian", "keyName": "E ionian" },
    { "nosSymbols": 5, "mode": "ionian", "keyName": "B ionian" },
    { "nosSymbols": 6, "mode": "ionian", "keyName": "F# ionian" },
    { "nosSymbols": 7, "mode": "ionian", "keyName": "C# ionian" },
    { "nosSymbols": 8, "mode": "ionian", "keyName": "G# ionian" },
    { "nosSymbols": 9, "mode": "ionian", "keyName": "D# ionian" },
    { "nosSymbols": 10, "mode": "ionian", "keyName": "A# ionian" },
    { "nosSymbols": 11, "mode": "ionian", "keyName": "E# ionian" },
    { "nosSymbols": -1, "mode": "ionian", "keyName": "F ionian" },
    { "nosSymbols": -2, "mode": "ionian", "keyName": "Bb ionian" },
    { "nosSymbols": -3, "mode": "ionian", "keyName": "Eb ionian" },
    { "nosSymbols": -4, "mode": "ionian", "keyName": "Ab ionian" },
    { "nosSymbols": -5, "mode": "ionian", "keyName": "Db ionian" },
    { "nosSymbols": -6, "mode": "ionian", "keyName": "Gb ionian" },
    { "nosSymbols": -7, "mode": "ionian", "keyName": "Cb ionian" },
    { "nosSymbols": -8, "mode": "ionian", "keyName": "Fb ionian" },
    { "nosSymbols": -9, "mode": "ionian", "keyName": "Bbb ionian" },
    { "nosSymbols": -10, "mode": "ionian", "keyName": "Ebb ionian" },
    { "nosSymbols": -11, "mode": "ionian", "keyName": "Abb ionian" },
    { "nosSymbols": 0, "mode": "dorian", "keyName": "D dorian" },
    { "nosSymbols": 1, "mode": "dorian", "keyName": "A dorian" },
    { "nosSymbols": 2, "mode": "dorian", "keyName": "E dorian" },
    { "nosSymbols": 3, "mode": "dorian", "keyName": "B dorian" },
    { "nosSymbols": 4, "mode": "dorian", "keyName": "F# dorian" },
    { "nosSymbols": 5, "mode": "dorian", "keyName": "C# dorian" },
    { "nosSymbols": 6, "mode": "dorian", "keyName": "G# dorian" },
    { "nosSymbols": 7, "mode": "dorian", "keyName": "D# dorian" },
    { "nosSymbols": 8, "mode": "dorian", "keyName": "A# dorian" },
    { "nosSymbols": 9, "mode": "dorian", "keyName": "E# dorian" },
    { "nosSymbols": 10, "mode": "dorian", "keyName": "B# dorian" },
    { "nosSymbols": 11, "mode": "dorian", "keyName": "F## dorian" },
    { "nosSymbols": -1, "mode": "dorian", "keyName": "G dorian" },
    { "nosSymbols": -2, "mode": "dorian", "keyName": "C dorian" },
    { "nosSymbols": -3, "mode": "dorian", "keyName": "F dorian" },
    { "nosSymbols": -4, "mode": "dorian", "keyName": "Bb dorian" },
    { "nosSymbols": -5, "mode": "dorian", "keyName": "Eb dorian" },
    { "nosSymbols": -6, "mode": "dorian", "keyName": "Ab dorian" },
    { "nosSymbols": -7, "mode": "dorian", "keyName": "Db dorian" },
    { "nosSymbols": -8, "mode": "dorian", "keyName": "Gb dorian" },
    { "nosSymbols": -9, "mode": "dorian", "keyName": "Cb dorian" },
    { "nosSymbols": -10, "mode": "dorian", "keyName": "Fb dorian" },
    { "nosSymbols": -11, "mode": "dorian", "keyName": "Bbb dorian" },
    { "nosSymbols": 0, "mode": "phrygian", "keyName": "E phrygian" },
    { "nosSymbols": 1, "mode": "phrygian", "keyName": "B phrygian" },
    { "nosSymbols": 2, "mode": "phrygian", "keyName": "F# phrygian" },
    { "nosSymbols": 3, "mode": "phrygian", "keyName": "C# phrygian" },
    { "nosSymbols": 4, "mode": "phrygian", "keyName": "G# phrygian" },
    { "nosSymbols": 5, "mode": "phrygian", "keyName": "D# phrygian" },
    { "nosSymbols": 6, "mode": "phrygian", "keyName": "A# phrygian" },
    { "nosSymbols": 7, "mode": "phrygian", "keyName": "E# phrygian" },
    { "nosSymbols": 8, "mode": "phrygian", "keyName": "B# phrygian" },
    { "nosSymbols": 9, "mode": "phrygian", "keyName": "F## phrygian" },
    { "nosSymbols": 10, "mode": "phrygian", "keyName": "C## phrygian" },
    { "nosSymbols": 11, "mode": "phrygian", "keyName": "G## phrygian" },
    { "nosSymbols": -1, "mode": "phrygian", "keyName": "A phrygian" },
    { "nosSymbols": -2, "mode": "phrygian", "keyName": "D phrygian" },
    { "nosSymbols": -3, "mode": "phrygian", "keyName": "G phrygian" },
    { "nosSymbols": -4, "mode": "phrygian", "keyName": "C phrygian" },
    { "nosSymbols": -5, "mode": "phrygian", "keyName": "F phrygian" },
    { "nosSymbols": -6, "mode": "phrygian", "keyName": "Bb phrygian" },
    { "nosSymbols": -7, "mode": "phrygian", "keyName": "Eb phrygian" },
    { "nosSymbols": -8, "mode": "phrygian", "keyName": "Ab phrygian" },
    { "nosSymbols": -9, "mode": "phrygian", "keyName": "Db phrygian" },
    { "nosSymbols": -10, "mode": "phrygian", "keyName": "Gb phrygian" },
    { "nosSymbols": -11, "mode": "phrygian", "keyName": "Cb phrygian" },
    { "nosSymbols": 0, "mode": "lydian", "keyName": "F lydian" },
    { "nosSymbols": 1, "mode": "lydian", "keyName": "C lydian" },
    { "nosSymbols": 2, "mode": "lydian", "keyName": "G lydian" },
    { "nosSymbols": 3, "mode": "lydian", "keyName": "D lydian" },
    { "nosSymbols": 4, "mode": "lydian", "keyName": "A lydian" },
    { "nosSymbols": 5, "mode": "lydian", "keyName": "E lydian" },
    { "nosSymbols": 6, "mode": "lydian", "keyName": "B lydian" },
    { "nosSymbols": 7, "mode": "lydian", "keyName": "F# lydian" },
    { "nosSymbols": 8, "mode": "lydian", "keyName": "C# lydian" },
    { "nosSymbols": 9, "mode": "lydian", "keyName": "G# lydian" },
    { "nosSymbols": 10, "mode": "lydian", "keyName": "D# lydian" },
    { "nosSymbols": 11, "mode": "lydian", "keyName": "A# lydian" },
    { "nosSymbols": -1, "mode": "lydian", "keyName": "Bb lydian" },
    { "nosSymbols": -2, "mode": "lydian", "keyName": "Eb lydian" },
    { "nosSymbols": -3, "mode": "lydian", "keyName": "Ab lydian" },
    { "nosSymbols": -4, "mode": "lydian", "keyName": "Db lydian" },
    { "nosSymbols": -5, "mode": "lydian", "keyName": "Gb lydian" },
    { "nosSymbols": -6, "mode": "lydian", "keyName": "Cb lydian" },
    { "nosSymbols": -7, "mode": "lydian", "keyName": "Fb lydian" },
    { "nosSymbols": -8, "mode": "lydian", "keyName": "Bbb lydian" },
    { "nosSymbols": -9, "mode": "lydian", "keyName": "Ebb lydian" },
    { "nosSymbols": -10, "mode": "lydian", "keyName": "Abb lydian" },
    { "nosSymbols": -11, "mode": "lydian", "keyName": "Dbb lydian" },
    { "nosSymbols": 0, "mode": "mixolydian", "keyName": "G mixolydian" },
    { "nosSymbols": 1, "mode": "mixolydian", "keyName": "D mixolydian" },
    { "nosSymbols": 2, "mode": "mixolydian", "keyName": "A mixolydian" },
    { "nosSymbols": 3, "mode": "mixolydian", "keyName": "E mixolydian" },
    { "nosSymbols": 4, "mode": "mixolydian", "keyName": "B mixolydian" },
    { "nosSymbols": 5, "mode": "mixolydian", "keyName": "F# mixolydian" },
    { "nosSymbols": 6, "mode": "mixolydian", "keyName": "C# mixolydian" },
    { "nosSymbols": 7, "mode": "mixolydian", "keyName": "G# mixolydian" },
    { "nosSymbols": 8, "mode": "mixolydian", "keyName": "D# mixolydian" },
    { "nosSymbols": 9, "mode": "mixolydian", "keyName": "A# mixolydian" },
    { "nosSymbols": 10, "mode": "mixolydian", "keyName": "E# mixolydian" },
    { "nosSymbols": 11, "mode": "mixolydian", "keyName": "B# mixolydian" },
    { "nosSymbols": -1, "mode": "mixolydian", "keyName": "C mixolydian" },
    { "nosSymbols": -2, "mode": "mixolydian", "keyName": "F mixolydian" },
    { "nosSymbols": -3, "mode": "mixolydian", "keyName": "Bb mixolydian" },
    { "nosSymbols": -4, "mode": "mixolydian", "keyName": "Eb mixolydian" },
    { "nosSymbols": -5, "mode": "mixolydian", "keyName": "Ab mixolydian" },
    { "nosSymbols": -6, "mode": "mixolydian", "keyName": "Db mixolydian" },
    { "nosSymbols": -7, "mode": "mixolydian", "keyName": "Gb mixolydian" },
    { "nosSymbols": -8, "mode": "mixolydian", "keyName": "Cb mixolydian" },
    { "nosSymbols": -9, "mode": "mixolydian", "keyName": "Fb mixolydian" },
    { "nosSymbols": -10, "mode": "mixolydian", "keyName": "Bbb mixolydian" },
    { "nosSymbols": -11, "mode": "mixolydian", "keyName": "Ebb mixolydian" },
    { "nosSymbols": 0, "mode": "aeolian", "keyName": "A aeolian" },
    { "nosSymbols": 1, "mode": "aeolian", "keyName": "E aeolian" },
    { "nosSymbols": 2, "mode": "aeolian", "keyName": "B aeolian" },
    { "nosSymbols": 3, "mode": "aeolian", "keyName": "F# aeolian" },
    { "nosSymbols": 4, "mode": "aeolian", "keyName": "C# aeolian" },
    { "nosSymbols": 5, "mode": "aeolian", "keyName": "G# aeolian" },
    { "nosSymbols": 6, "mode": "aeolian", "keyName": "D# aeolian" },
    { "nosSymbols": 7, "mode": "aeolian", "keyName": "A# aeolian" },
    { "nosSymbols": 8, "mode": "aeolian", "keyName": "E# aeolian" },
    { "nosSymbols": 9, "mode": "aeolian", "keyName": "B# aeolian" },
    { "nosSymbols": 10, "mode": "aeolian", "keyName": "F## aeolian" },
    { "nosSymbols": 11, "mode": "aeolian", "keyName": "C## aeolian" },
    { "nosSymbols": -1, "mode": "aeolian", "keyName": "D aeolian" },
    { "nosSymbols": -2, "mode": "aeolian", "keyName": "G aeolian" },
    { "nosSymbols": -3, "mode": "aeolian", "keyName": "C aeolian" },
    { "nosSymbols": -4, "mode": "aeolian", "keyName": "F aeolian" },
    { "nosSymbols": -5, "mode": "aeolian", "keyName": "Bb aeolian" },
    { "nosSymbols": -6, "mode": "aeolian", "keyName": "Eb aeolian" },
    { "nosSymbols": -7, "mode": "aeolian", "keyName": "Ab aeolian" },
    { "nosSymbols": -8, "mode": "aeolian", "keyName": "Db aeolian" },
    { "nosSymbols": -9, "mode": "aeolian", "keyName": "Gb aeolian" },
    { "nosSymbols": -10, "mode": "aeolian", "keyName": "Cb aeolian" },
    { "nosSymbols": -11, "mode": "aeolian", "keyName": "Fb aeolian" },
    { "nosSymbols": 0, "mode": "locrian", "keyName": "B locrian" },
    { "nosSymbols": 1, "mode": "locrian", "keyName": "F# locrian" },
    { "nosSymbols": 2, "mode": "locrian", "keyName": "C# locrian" },
    { "nosSymbols": 3, "mode": "locrian", "keyName": "G# locrian" },
    { "nosSymbols": 4, "mode": "locrian", "keyName": "D# locrian" },
    { "nosSymbols": 5, "mode": "locrian", "keyName": "A# locrian" },
    { "nosSymbols": 6, "mode": "locrian", "keyName": "E## locrian" },
    { "nosSymbols": 7, "mode": "locrian", "keyName": "B## locrian" },
    { "nosSymbols": 8, "mode": "locrian", "keyName": "F## locrian" },
    { "nosSymbols": 9, "mode": "locrian", "keyName": "C## locrian" },
    { "nosSymbols": 10, "mode": "locrian", "keyName": "G## locrian" },
    { "nosSymbols": 11, "mode": "locrian", "keyName": "D## locrian" },
    { "nosSymbols": -1, "mode": "locrian", "keyName": "E locrian" },
    { "nosSymbols": -2, "mode": "locrian", "keyName": "A locrian" },
    { "nosSymbols": -3, "mode": "locrian", "keyName": "D locrian" },
    { "nosSymbols": -4, "mode": "locrian", "keyName": "G locrian" },
    { "nosSymbols": -5, "mode": "locrian", "keyName": "C locrian" },
    { "nosSymbols": -6, "mode": "locrian", "keyName": "F locrian" },
    { "nosSymbols": -7, "mode": "locrian", "keyName": "Bb locrian" },
    { "nosSymbols": -8, "mode": "locrian", "keyName": "Eb locrian" },
    { "nosSymbols": -9, "mode": "locrian", "keyName": "Ab locrian" },
    { "nosSymbols": -10, "mode": "locrian", "keyName": "Db locrian" },
    { "nosSymbols": -11, "mode": "locrian", "keyName": "Gb locrian"}
  ];

  function nos_symbols_and_mode2key_name(nos_symbols, mode){
    // Tom Collins 19/2/2015.
    // In
    // nos_symbols Integer mandatory
    // mode String mandatory
    // Out String
  	// This function takes the number of symbols in a key signature and a string
    // specifying the mode, and converts these pieces of information to a string
    // naming the key signature. For instance, -2 symbols means 2 flats, and
    // aeolian mode would give G aeolian.


    var i = 0;
  	while (i < keysLookup.length){
  		if (keysLookup[i].nosSymbols == nos_symbols &&
  				keysLookup[i].mode == mode){
  			var key_name = keysLookup[i].keyName;
        i = keysLookup.length - 1;
  		}
      i++;
  	}
  	if (key_name == undefined){
  		return "not specified";
  	}
  	else {
  		return key_name;
  	}
  }

  function row_of_max_bar_leq_bar_arg(bar, time_sigs_array){
    // Tom Collins 17/10/2014.
    // In
    // bar Integer mandatory
    // time_sigs_array Array mandatory
    // Out Array
    // This function returns the row (in a list of time signatures) of the
    // maximal bar number less than or equal to the bar number argument.

    var bar_out = time_sigs_array[0];
    var i = 0;
    var n = time_sigs_array.length;
    while (i < n) {
      if (bar < time_sigs_array[i]["barNo"]){
        bar_out = time_sigs_array[i - 1];
        i = n - 1;
      }
      else if (bar == time_sigs_array[0]["barNo"]){
        bar_out = time_sigs_array[i];
        i = n - 1;
      }
      else if (i == n - 1){
        bar_out = time_sigs_array[i];
      }
      i=i+1;
    }
    return bar_out;
  }

  /**
   * Given a bar number and beat number, and a time-signature array (with ontimes
   * appended), this function returns the ontime of that bar and beat number.
   *
   * @author Tom Collins
   * @comment 17th October 2014
   * @param {number} bar - A bar number for which we want to know the
   * corresponding ontime (time counting in crotchet beats starting from 0 for bar
   * 1 beat 1).
   * @param {number} beat - A beat number for which we want to know the
   * corresponding ontime (time counting in crotchet beats starting from 0 for bar
   * 1 beat 1).
   * @param {TimeSignature[]} time_sigs_array - An array of time signatures.
   * @return {number} An ontime
   *
   * @example
   *     var tsArr = [
   *   {
   *     "barNo": 1,
   *     "topNo": 4,
   *     "bottomNo": 4,
   *     "ontime": 0
   *   },
   *   {
   *     "barNo": 3,
   *     "topNo": 3,
   *     "bottomNo": 4,
   *     "ontime": 8
   *   }
   * ];
   * ontime_of_bar_and_beat_number(4, 1, tsArr);
   * â†’
   * 11
   */
   function ontime_of_bar_and_beat_number(bar, beat, time_sigs_array){
    var n = time_sigs_array.length;
    var relevant_row = row_of_max_bar_leq_bar_arg(bar, time_sigs_array);
    var excess = bar - relevant_row["barNo"];
    var local_beat_bar = relevant_row["topNo"]*4/relevant_row["bottomNo"];
    var a = relevant_row["ontime"] + excess*local_beat_bar + beat - 1;
    return a;
  }

  const pitch_class_lookup_array$1 = [
    [[12, 6], "B#"], [[0, 0], "C"], [[0, 1], "Dbb"],
    [[13, 6], "B##"], [[1, 0], "C#"], [[1, 1], "Db"],
    [[2, 0], "C##"], [[2, 1], "D"], [[2, 2], "Ebb" ],
    [[3, 1], "D#"], [[3, 2], "Eb"], [[3, 3], "Fbb"],
    [[4, 1], "D##"], [[4, 2], "E"], [[4, 3], "Fb"],
    [[5, 2], "E#"], [[5, 3], "F"], [[5, 4], "Gbb"],
    [[6, 2], "E##"], [[6, 3], "F#"], [[6, 4], "Gb"],
    [[7, 3], "F##"], [[7, 4], "G"], [[7, 5], "Abb"],
    [[8, 4], "G#"], [[8, 5], "Ab"],
    [[9, 4], "G##"], [[9, 5], "A"], [[9, 6], "Bbb"],
    [[-2, 0], "Cbb"], [[10, 5], "A#"], [[10, 6], "Bb"],
    [[11, 5], "A##"], [[11, 6], "B"], [[-1, 0], "Cb"]
  ];

  function pitch_and_octave2midi_note_morphetic_pair(pitch_and_octave){
    // Tom Collins 15/10/2014.
    // In
    // pitch_and_octave String mandatory
    // Out Array
    // This function converts a string consisting of a note's pitch and octave
    // into a  pair consisting of a MIDI note number and a morphetic pitch
    // number.

    var length_arg = pitch_and_octave.length;
    var pitch_class = pitch_and_octave.slice(0, length_arg - 1);
    var octave = pitch_and_octave[length_arg - 1];

    var pitch_class_idx = 1;
    var n = pitch_class_lookup_array$1.length;
    var i = 0;
    while (i < n){
      if (pitch_class == pitch_class_lookup_array$1[i][1]){
        pitch_class_idx = i;
        i = n - 1;
      }
      i=i+1;
    }
    var midi_mpn_residue = pitch_class_lookup_array$1[pitch_class_idx][0];
    var a = [];
    a[0] = 12*octave + 12 + midi_mpn_residue[0];
    a[1] = 7*octave + 32 + midi_mpn_residue[1];
    return a;
  }

  function remove_duplicate_clef_changes(clef_changes){
    // Tom Collins 23/2/2015.
    // In
    // clef_changes Array mandatory
    // Out Array
    // This function inspects pairs of clef changes. If there is a clef change
    // in bar n, and a clef change to the same clef in bar n + 1, the clef
    // change in bar n is removed because it is probably a cautionary.

    var arr_out = [];
    for (let clefi = 0; clefi < clef_changes.length - 1; clefi++){
      if (clef_changes[clefi + 1].barNo != clef_changes[clefi].barNo + 1 ||
          clef_changes[clefi + 1].clef != clef_changes[clefi].clef ||
          clef_changes[clefi + 1].staffNo != clef_changes[clefi].staffNo){
        arr_out.push(clef_changes[clefi]);
      }
    }
    if (clef_changes.length > 0){
      arr_out.push(clef_changes[clef_changes.length - 1]);
    }
    return arr_out;
  }

  function resolve_expressions(expressions){
    // Tom Collins 28/2/2015
    // In
    // expressions Array mandatory
    // Out Array
    // When crescendos and diminuendos are expressed as lines (hairpins, wedges),
    // they have a stopping point as well as a starting point. This function
    // locates wedges stops corresponding to wedge starts, and unites the two
    // pieces of information in one array object.

    // Remove all stop wedges from the expressions array.
    var wedge_stops = [];
    var i = expressions.length - 1;
    while (i >= 0){
      if (expressions[i].type.wedge && expressions[i].type.wedge == "stop"){
        wedge_stops.push(expressions[i]);
        expressions.splice(i, 1);
      }
      i--;
    }
    // Loop over the expressions array and associate each wedge with a member of
    // wedge_stops.
    let target_idx;
    for (let j = 0; j < expressions.length; j++){
      if (expressions[j].type.wedge){
        // Find the target index in wedge_stops.
        target_idx = -1;
        var k = 0;
        while (k < wedge_stops.length){
          if (wedge_stops[k].staffNo == expressions[j].staffNo &&
              wedge_stops[k].placement == expressions[j].placement &&
              wedge_stops[k].ontime >= expressions[j].ontime){
            // We found it!
            target_idx = k;
            k = wedge_stops.length - 1;
          }
          k++;
        }
        if (target_idx >= 0){
          // Add some properties to expressions[j].
          expressions[j].barOff = wedge_stops[target_idx].barOn;
          expressions[j].beatOff = wedge_stops[target_idx].beatOn;
          expressions[j].offtime = wedge_stops[target_idx].ontime;
        }
        else {
          console.log('Could not find a stop for wedge: ', expressions[j]);
        }
      }
    }
    return expressions;
  }

  function sort_points_asc(a, b){
    // Tom Collins 17/11/2014.
    // In
    // a Object mandatory
    // b Object mandatory
    // Out Object
  	// A helper function to sort two notes (points) or rests by ascending ontime.
  	// If the ontimes match and MNNs are defined, sort by these instead. If these
  	// match, sort by staffNo. If these match, sort by voiceNo.

  	if (a.ontime != b.ontime){
      return a.ontime - b.ontime;
  	}
  	if (a.MNN != undefined){
  		if (a.MNN != b.MNN){
  			return a.MNN - b.MNN;
  		}
  	}
  	if (a.staffNo != b.staffNo){
  		return a.staffNo - b.staffNo;
  	}
  	return a.voiceNo - b.voiceNo;
  }

  function sort_points_asc_by_id$1(a, b){
    // Tom Collins 18/2/2015.
    // In
    // a Object mandatory
    // b Object mandatory
    // Out Object
  	// A helper function, to sort two notes (points) or rests ascending by the
    // values in the id field.

    var c = a.ID;
    var d = b.ID;
    if (typeof c == "string"){
      c = parseFloat(c);
    }
    if (typeof d == "string"){
      d = parseFloat(d);
    }
  	return c - d;
  }

  function staff_voice_xml2staff_voice_json(voice_no_from_xml, staff_nos_for_this_id, part_idx){
    // Tom Collins 22/2/2015.
    // In
    // voice_no_from_xml Integer mandatory
    // staff_nos_for_this_id Array mandatory
    // part_idx Integer mandatory
    // Out Array
  	// This function converts MusicXML 2.0 voice assignments, which can go beyond
    // 1-4 into 5-8 in order to encode multiple staves within the same part, to
    // json_score voice assignments, which use staff number to encode multiple
    // staves within the same part separately, and a voice number always in the
    // range 0-3.

    if (voice_no_from_xml !== undefined){
      // There is a maximum of four voices per staff. In MusicXML 2.0, voices 5-8
      // are used to encode a second staff in the same part. In a json_score
      // these will have separate staff numbers, and this is handled here. The
      // convention of using voices 5-8 to encode a second staff in the same part
      // is not adhered to by hum2xml.
      var staff_idx = Math.floor((voice_no_from_xml - 1)/4);
      var staffNo = staff_nos_for_this_id[staff_idx];
      var voiceNo = voice_no_from_xml%4 - 1;
    }
    else {
      var staffNo = part_idx;
      var voiceNo = 0;
    }
    return [staffNo, voiceNo];
  }

  // File conversion.
  function comp_obj2note_point_set(comp_obj){
    // Tom Collins 2/2/2015.
    // In
    // comp_obj Object mandatory
    // Out Array
    // This function iterates over the notes property of a Composition object,
    // and converts the objects found there into a point-set format, with
    // so-called columns for ontime, MNN, MPN, duration, staff number, and
    // velocity in [0, 1].

    var notes = comp_obj.notes;
    var out_array = [];
    for (let inote = 0; inote < notes.length; inote++){
      var note = [
        notes[inote].ontime,
        notes[inote].MNN,
        notes[inote].MPN,
        notes[inote].duration,
        notes[inote].staffNo
      ];
      if (notes[inote].tonejs !== undefined && notes[inote].tonejs.volume !== undefined) {
        note.push(notes[inote].tonejs.volume);
      } else {
        note.push(.8);
      }
      out_array.push(note);
    }
    return out_array;
  }

  function restrict_point_set_in_nth_to_xs(point_set, n, xs){
    // Tom Collins 24/11/2014.
    // In
    // point_set Array mandatory
    // n Integer mandatory
    // xs Array mandatory
    // Out Array
    // The first argument to this function is an array consisting of numeric
    // arrays of uniform dimension (what I call a point set). We are interested
    // in the nth element of each array, where n is the second argument. A point
    // is retained in the output if its nth value is a member of the array
    // specified by the third argument.

    var point_set_out = [];
    for (let ip = 0; ip < point_set.length; ip++){
      if (xs.indexOf(point_set[ip][n]) != -1){
        point_set_out.push(point_set[ip]);
      }
    }
    return point_set_out;
  }

  function get_unique(arr){
    // Tom Collins 24/11/2014.
    // In
    // arr Array mandatory
    // Out Array
    // This function returns unique elements from the input array. It will not
    // handle nested arrays properly (see unique_rows).

    var a = [];
    for (let i=0, l=arr.length; i<l; i++){
      if (a.indexOf(arr[i]) === -1){
        a.push(arr[i]);
      }
    }
    return a;
  }

  function split_point_set_by_staff(point_set, staff_idx){
    // Tom Collins 2/2/2015.
    // In
    // point_set Array mandatory
    // staff_idx Integer mandatory
    // Out Array
    // This function splits a point set into multiple point sets, grouping by the
    // values in the (staff_idx)th element.

    // Get the unique staves.
    var staves = [];
    for (let ipt = 0; ipt < point_set.length; ipt++){
      staves.push(point_set[ipt][staff_idx]);
    }
    var unq_staves = get_unique(staves).sort(function(a, b){return a - b});
    var out_array = [];
    // Create a point set consisting of points in each staff.
    for (let iuq = 0; iuq < unq_staves.length; iuq++){
      var curr_point_set = restrict_point_set_in_nth_to_xs(
        point_set, staff_idx, [unq_staves[iuq]]);
      out_array[iuq] = curr_point_set;

    }
    return out_array;
  }

  function copy_array_object(arr){
    // Tom Collins 21/2/2015.
    // In
    // arr Array mandatory
    // Out Array
    // This function returns an independent copy of an array object.

    return JSON.parse(JSON.stringify(arr));
  }

  // Point-set operations.
  function copy_point_set(point_set){
    // Tom Collins 24/11/2014.
    // In
    // point_set Array mandatory
    // Out Array
    // This function returns an independent copy of a point set.

    var E = [];
    point_set.map(function(x){
      E.push(x.slice());
    });
    return E;

    // Old version.
    // var n = point_set.length;
    // var E = new Array(n);
    // var i = 0; // Increment over D and E.
    // while (i < n){
    //   E[i] = point_set[i].slice();
    //   i++;
    // }
    // return E;
  }

  function index_point_set(point_set){
    // Tom Collins 24/11/2014.
    // In
    // point_set Array mandatory
    // Out Array
    // This function pushes index values to the last element of each point.

    var k = point_set[0].length;
    var n = point_set.length;
    var i = 0; // Increment over point_set.
    while (i < n){
      point_set[i][k] = i;
      i++;
    }
    return point_set;
  }

  function lex_more(u, v, k){
    // Tom Collins 24/11/2014.
    // In
    // u Array mandatory
    // v Array mandatory
    // k Integer optional
    // This function returns 1 if u is more than v, where more than is the
    // lexicographic ordering. It returns -1 otherwise.

    // In general, for two vectors u and v, this function finds the first index
    // i such that u(i) is not equal to v(i). If u(i) is more than v(i), then u
    // is more than v. If v(i) is more than u(i), then v is more than u. In
    // the event that u equals v, u is not more than v.

    if (typeof k === 'undefined') {
      k = u.length;
    }
    // Logical outcome.
    var tf = -1;
    var i = 0; // Increment over u, v.
    while (i < k){
      if (u[i] == v[i]){
        i++;
      }
      else {
        if (u[i] > v[i]){
          tf = 1;
          i = k + 1;
        }
        else {
          i = k + 1;
        }
      }
    }
    return tf;
  }

  function sort_rows(point_set){
    // Tom Collins 24/11/2014.
    // In
    // point_set Array mandatory
    // Out Array
    // The only argument to this function is an array consisting of numeric
    // arrays of uniform dimension (what I call a point set). This function
    // returns the elements in lexicographic order as first argument. As second
    // argument are the indices of each element from the input array.

    // Create an independent copy of the dataset.
    var E = copy_point_set(point_set);
    // Index the copied dataset.
    E = index_point_set(E);
    // Sort the indexed and copied dataset.
    E.sort(lex_more);
    // Create a new variable that will contain just the dataset.
    var k = point_set[0].length;
    var n = point_set.length;
    var F = new Array(n);
    // Create a new variable that will contain just the index.
    var g = new Array(n);
    var i = 0; // Increment over E, F, and g.
    while (i < n){
      F[i] = E[i].slice(0, k);
      g[i] = E[i][k];
      i++;
    }
    return [F, g];
  }

  /**
   * This function counts rows of the input `point_set`, weighted, if desired, by
   * values in `wght_idx`.
   *
   * @author Tom Collins and Christian Coulon
   * @comment 7th November 2015
   * @param {PointSet} point_set - A point set
   * @param {number} [wght_idx] - The dimension of each point that should be used
   * to weight the count. If left undefined, each occurrence of a point will
   * increment the count of that point by 1.
   * @return {Object} [PointSet, number[]] An array whose first element is a
   * {@link PointSet} (unique and lexicographically ascending version of the
   * input), and whose second element is a (possibly weighted) count of those
   * points in the input.
   *
   * @example
   *     var ps = [[64, 2], [65, 1], [67, 1], [67, 1]];
   * var w = 1;
   * count_rows(ps, w)
   * â†’
   * [
   *   [
   *     [64, 2], [65, 1], [67, 1]
   *   ],
   *   [
   *     2, // One occurrence of [64, 2] in input point set, with weight 2.
   *     1, // One occurrence of [65, 1] in input point set, with weight 1.
   *     2 // Two occurrences of [67, 1] in input point set, each with weight 1.
   *   ]
   * ]
   */
  function count_rows(point_set, wght_idx){
    // No check on point_set credentials at present...
    if (wght_idx !== undefined && wght_idx < point_set[0].length){
      // Make a copy of the point_set, where wght_idx values are in the final
      // dimension of each point.
      var arr = copy_point_set(point_set);
      var arr2 = [];
      for (let i = 0; i < arr.length; i++){
        var curr_rmv = point_set[i][wght_idx];
        // console.log('curr_rmv:');
        // console.log(curr_rmv);
        arr[i].splice(wght_idx, 1);
        // console.log('arr[i]:');
        // console.log(arr[i]);
        arr2.push(arr[i].concat([curr_rmv]));
      }
      // Sort the rows of a copy of the dataset.
      var E = sort_rows(arr2);
      var F = E[0];
      // var g = E[1];
      // Create a new variable that will contain the unique rows of the dataset.
      var k = point_set[0].length - 1;
      var U = [];
      // Create a new variable that will contain the count of each unique row in
      // the original dataset.
      var v = [];
      U[0] = F[0].slice(0, k);
      v[0] = F[0][k];
      var i = 1; // Increment over F and g.
      var j = 1; // Increment over U and v.
      while (i < point_set.length){
        if (array_equals(F[i].slice(0, k),F[i - 1].slice(0, k))){
          v[j - 1] = v[j - 1] + F[i][k];
        }
        else {
          U[j] = F[i].slice(0, k);
          v[j] = F[i][k];
          j++;
        }
        i++;
      }
    }
    else {
      // Sort the rows of a copy of the dataset.
      var E = sort_rows(point_set);
      var F = E[0];
      // var g = E[1];
      // Create a new variable that will contain the unique rows of the dataset.
      var k = point_set[0].length;
      var U = [];
      // Create a new variable that will contain the count of each unique row in
      // the original dataset.
      var v = [];
      U[0] = F[0];
      v[0] = 1;
      var i = 1; // Increment over F and g.
      var j = 1; // Increment over U and v.
      while (i < point_set.length){
        if (array_equals(F[i],F[i - 1])){
          v[j - 1]++;
        }
        else {
          U[j] = F[i];
          v[j] = 1;
          j++;
        }
        i++;
      }
    }
    return [U.slice(0, j), v.slice(0, j)];
  }

  function mean(arr){
    // Christian Coulon and Tom Collins 17/10/2014.
    // In
    // arr Array mandatory
    // Out Number
    // This function returns the mean of an input numeric array.

    if (!arr.length){
      return 0;
    }
    else {
      var sum = 0;
      for (let i = 0; i < arr.length; i++){
        sum += arr[i];
      }
      return sum/arr.length;
    }
  }

  function min_argmin (arr){
    // Tom Collins 21/10/2014.
    // In
    // arr Array mandatory
    // Out Array
    // Returns the minimum element in an array and its index (argument).

    var min = arr[0];
    var minIndex = 0;
    for (let i = 1; i < arr.length; i++) {
      if (arr[i] < min) {
        minIndex = i;
        min = arr[i];
      }
    }
    return [min, minIndex];

    // CDC said the following is the same, but it does not retain the index of
    // the minimum element:
    // return arr.reduce(function(a, b){ return a < b?a:b; }, arr[0]);
  }

  function tonic_pitch_closest(points, key_name){
    // Tom Collins 22/1/2016.
    // In
    // points Array mandatory
    // key_name String mandatory
    // Out Array
    // This function returns the MIDI note and morphetic pitch numbers of the tonic
    // pitch that is closest to the mean of the input point set, whose key
    // (estimate) is specified as the second input argument.

    // Get the pitch class.
    var pitch_class = key_name.split(" ")[0];
    // Create an array of MNN-MPN pairs for this pitch class across the octaves.
    var min_idx = 1;
    if (pitch_class == "A" || pitch_class == "Bb" || pitch_class == "B"){
      min_idx = 0;
    }
    var max_idx = 7;
    if (pitch_class == "C"){
      max_idx = 8;
    }
    var MNN_MPNs = [];
    for (let i = min_idx; i <= max_idx; i++){
      var curr_pitch = pitch_class + i.toString();
      MNN_MPNs.push(pitch_and_octave2midi_note_morphetic_pair(curr_pitch));
    }

    var mu = mean(points.map(function(a){
      return a[1];
    }));
    var mnn_mu = MNN_MPNs.map(function(a){
      return Math.abs(a[0] - mu);
    });
    min_idx = min_argmin(mnn_mu);
    return MNN_MPNs[min_idx[1]];
  }

  function max_argmax(arr){
    // Tom Collins 21/10/2014.
    // In
    // arr Array mandatory
    // Out Array
    // Returns the maximum element in an array and its index (argument).

    var max = arr[0];
    var maxIndex = 0;
    for (let i = 1; i < arr.length; i++) {
      if (arr[i] > max) {
        maxIndex = i;
        max = arr[i];
      }
    }
    return [max, maxIndex];

    // CDC said the following is the same, but it does not retain the index of
    // the maximum element:
    // return arr.reduce(function(a, b){ return a > b?a:b; }, arr[0]);
  }

  function corr(x, y){
    // Tom Collins 8/11/2015.
    // In
    // x Array mandatory
    // y Array mandatory
    // Out Number
    // This function calculates the Pearson product-moment correlation
    // coefficient between the input arrays x and y. It checks that the arrays
    // are of the same length, but does not check that they each consist of
    // numbers, nor for zero divisors (output NaN in both cases).

    var n = x.length;
    if (n !== y.length){
      throw "Error in call to corr: input arrays must be of the same length.";
    }
    else {
      var x_bar = mean(x);
      var y_bar = mean(y);
      var x2 = 0;
      var y2 = 0;
      var xy = 0;
      for (let i = 0; i < x.length; i++){
        x2 += Math.pow(x[i], 2);
        y2 += Math.pow(y[i], 2);
        xy += x[i]*y[i];
      }
      var r = (xy - n*x_bar*y_bar)/
        (Math.sqrt(x2 - n*Math.pow(x_bar, 2))*Math.sqrt(y2 - n*Math.pow(y_bar, 2)));
      return r;
    }
  }

  function cyclically_permute_array_by(arr, m){
    // Tom Collins 6/11/2015.
    // In
    // arr Array mandatory
    // m Non-negative integer mandatory
    // Out Array
    // This function moves the ith element of an array (counting from zero) to
    // the zeroth element in the output array, where i is the second argument.
    // The (i - 1)th element is moved to the last element in the output array,
    // the (i - 2)th element is moved to the penultimate element in the output
    // array, etc.

    m = m % arr.length;
    var arr2 = copy_array_object(arr);
    var arr3 = arr2.slice(0, m);
    var arr4 = arr2.slice(m).concat(arr3);
    return arr4;
  }

  function orthogonal_projection_not_unique_equalp(point_set, indicator){
    // Tom Collins 22/12/2014.
    // In
    // point_set Array mandatory
    // indicator Array mandatory
    // Out Array
    // Given a set of vectors (all members of the same n-dimensional vector
    // space), and an n-tuple of zeros and ones indicating a particular
    // orthogonal projection, this function returns the projected set of vectors.

    var set_out = [];
    for (let ip = 0; ip < point_set.length; ip++){
      var curr_point = [];
      for (let id = 0; id < point_set[0].length; id++){
        if (indicator[id] == 1){
          curr_point.push(point_set[ip][id]);
        }
      }
      set_out.push(curr_point);
    }
    return set_out;
  }

  // Keyscape.
  // Setup the key profiles.
  var key_names = [
    "C major", "Db major", "D major", "Eb major", "E major", "F major",
    "Gb major", "G major", "Ab major", "A major", "Bb major", "B major",
    "C minor", "C# minor", "D minor", "Eb minor", "E minor", "F minor",
    "F# minor", "G minor", "G# minor", "A minor", "Bb minor", "B minor"];

  const aarden_key_profiles = {};
  for (let ikey = 0; ikey < 12; ikey++){
    aarden_key_profiles[key_names[ikey]] =
      cyclically_permute_array_by(
      [
        17.77, 0.15, 14.93, 0.16, 19.8, 11.36, 0.29, 22.06, 0.15, 8.15, 0.23,
        4.95
      ],
      -ikey
    );
  }
  for (let ikey = 12; ikey < 24; ikey++){
    aarden_key_profiles[key_names[ikey]] =
      cyclically_permute_array_by(
      [
        18.26, 0.74, 14.05, 16.86, 0.7, 14.44, 0.7, 18.62, 4.57, 1.93, 7.38, 1.76
      ],
      -ikey
    );
  }

  const krumhansl_and_kessler_key_profiles = {};
  for (let ikey = 0; ikey < 12; ikey++){
    krumhansl_and_kessler_key_profiles[key_names[ikey]] =
      cyclically_permute_array_by(
      [
        6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88
      ],
      -ikey
    );
  }
  for (let ikey = 12; ikey < 24; ikey++){
    krumhansl_and_kessler_key_profiles[key_names[ikey]] =
      cyclically_permute_array_by(
      [
        6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17
      ],
      -ikey
    );
  }

  /**
   * This function is an implementation of the Krumhansl-Schmuckler key-finding
   * algorithm. It returns a key estimate for the input points.
   *
   * @author Tom Collins and Christian Coulon
   * @comment 6th November 2015
   * @tutorial key-estimation-1
   * @param {PointSet} point_set - A point set
   * @param {number[][]} key_profiles - An array of arrays of 12-element vectors,
   * where each vector is a cylic permutation of its neighbour. These are
   * empirically derived weightings of the pitch-class content of each of the
   * major and minor keys. There is a choice between
   * `krumhansl_and_kessler_key_profiles` and `aarden_key_profiles`, which have
   * different weightings.
   * @param {number} [MNN_idx=1] - The dimension of MIDI note numbers in `ps`
   * @param {number} [dur_idx=2] - The dimension of durations in `ps`
   * @return {Object} [string, number, number, number] An array whose first
   * element is a key name (e.g., "C major" for C major), whose second element is
   * the maximum correlation value, whose third element is steps on the circle of
   * fifths (e.g., -1 for F major), and whose fourth element is mode (e.g., 5 for
   * minor/Aeolian). So for instance, a key estimate of A minor might have output
   * `["A minor", .8, 3, 5]`.
   *
   * @example
   *     var ps = [
   *   [0, 56, 1],
   *   [0.5, 60, 1],
   *   [1, 58, 1],
   *   [1.5, 61, 1],
   *   [2, 60, 1],
   *   [2.5, 63, 1],
   *   [3, 61, 1],
   *   [3.5, 65, 1],
   *   [4, 63, 1],
   *   [4.5, 67, 1],
   *   [5, 65, 1],
   *   [5.5, 68, 1],
   *   [6, 67, 1],
   *   [6.5, 70, 1],
   *   [7, 68, 2]
   * ];
   * fifth_steps_mode(ps, krumhansl_and_kessler_key_profiles)
   * â†’
   * [
   *   "Ab major", // Estimated key
   *   0.90135,    // Winning (maximal) correlation
   *   -4,         // Steps on the circle of fifths for Ab
   *   0           // Mode (major/Ionian)
   * ]
   */
  function fifth_steps_mode(point_set, key_profiles, MNN_idx, dur_idx){
    if (MNN_idx == undefined){
      MNN_idx = 1;
    }
    if (dur_idx == undefined){
      dur_idx = 2;
    }

    // Copy the point_set variable.
    var pts = copy_array_object(point_set);
    // Convert the MNNs to MNNs mod 12.
    for (let i = 0; i < pts.length; i++){
      pts[i][MNN_idx] = pts[i][MNN_idx] % 12;
    }
    // Get the MNN12s and durations in an array. Begin by constructing the
    // indicator to pass to orthogonal_projection_not_unique_equalp.
    var indicator = [];
    for (let i = 0; i < pts[0].length; i++){
      if (i == MNN_idx || i == dur_idx){
        indicator[i] = 1;
      }
      else {
        indicator[i] = 0;
      }
    }
    var arr = orthogonal_projection_not_unique_equalp(pts, indicator);
    let wght_idx;
    // Form a distribution over the MNN12s, weighted by duration.
    if (dur_idx >= MNN_idx){
      wght_idx = 1;
    }
    else {
      wght_idx = 0;
    }
    var MNN12_and_distbn = count_rows(arr, wght_idx);
    // Convert to a key profile.
    var idxs = [];
    for (let i = 0; i < MNN12_and_distbn[0].length; i++){
      idxs.push(MNN12_and_distbn[0][i][0]);
    }
    var kp = [];
    for (let i = 0; i < 12; i++){
      kp[i] = 0;
    }
    for (let i = 0; i < idxs.length; i++){
      kp[idxs[i]] = MNN12_and_distbn[1][i];
    }

    // Calculate the correlation between the empirical key profile and each of
    // the theoretical key profiles.
    var key_names = Object.keys(key_profiles);
    var r = [];
    for (let i = 0; i < 24; i++){
      r[i] = corr(kp, key_profiles[key_names[i]]);
    }

    // Prepare the return in terms of fith steps and mode.
    var corr_and_key = max_argmax(r);
    var quasi_key = corr_and_key[1];
    var steps = [0, -5, 2, -3, 4, -1, 6, 1, -4, 3, -2, 5];
    var mode = 0;
    if (quasi_key >= 12){
      mode = 5;
    }
    return [key_names[quasi_key], corr_and_key[0], steps[quasi_key % 12], mode];
  }

  // Chord labeling.
  const chord_templates_pbmin7ths = [
    // Major triads
    [0, 4, 7], [1, 5 ,8], [2, 6, 9], [3, 7, 10], [4, 8, 11], [5 ,9, 0],
    [6, 10, 1], [7, 11, 2], [8, 0, 3], [9, 1, 4], [10, 2, 5], [11, 3, 6],
    // Dominant 7th triads
    [0, 4, 7, 10], [1, 5 ,8, 11], [2, 6, 9, 0], [3, 7, 10, 1],
    [4, 8, 11, 2], [5 ,9, 0, 3], [6, 10, 1, 4], [7, 11, 2, 5],
    [8, 0, 3, 6], [9, 1, 4, 7], [10, 2, 5 ,8], [11, 3, 6, 9],
    // Minor triads
    [0, 3, 7], [1, 4, 8], [2, 5 ,9], [3, 6, 10], [4, 7, 11], [5 ,8, 0],
    [6, 9, 1], [7, 10, 2], [8, 11, 3], [9, 0, 4], [10, 1, 5], [11, 2, 6],
    // Fully diminished 7th
    [0, 3, 6, 9], [1, 4, 7, 10], [2, 5 ,8, 11],
    // Half diminished 7th
    [0, 3, 6, 10], [1, 4, 7, 11], [2, 5 ,8, 0], [3, 6, 9, 1],
    [4, 7, 10, 2], [5 ,8, 11, 3], [6, 9, 0, 4], [7, 10, 1, 5],
    [8, 11, 2, 6], [9, 0, 3, 7], [10, 1, 4, 8], [11, 2, 5 ,9],
    // Diminished triad
    [0, 3, 6], [1, 4, 7], [2, 5 ,8], [3, 6, 9], [4, 7, 10], [5 ,8, 11],
    [6, 9, 0], [7, 10, 1], [8, 11, 2], [9, 0, 3], [10, 1, 4],
    [11, 2, 5],
    // Minor 7th
    [0, 3, 7, 10], [1, 4, 8, 11], [2, 5 ,9, 0], [3, 6, 10, 1],
    [4, 7, 11, 2], [5 ,8, 0, 3], [6, 9, 1, 4], [7, 10, 2, 5],
    [8, 11, 3, 6], [9, 0, 4, 7], [10, 1, 5 ,8], [11, 2, 6, 9]
  ];

  const chord_lookup_pbmin7ths = [
    "C major", "Db major", "D major", "Eb major", "E major", "F major",
    "F# major", "G major", "Ab major", "A major", "Bb major", "B major",
    "C 7", "Db 7", "D 7", "Eb 7", "E 7", "F 7",
    "F# 7", "G 7", "Ab 7", "A 7", "Bb 7", "B 7",
    "C minor", "Db minor", "D minor", "Eb minor", "E minor", "F minor",
    "F# minor", "G minor", "Ab minor", "A minor", "Bb minor", "B minor",
    // Because Pardo & Birmingham (2002) only use MIDI note,
    // there is a bit of an issue with diminished 7th chords
    // (next three labels), as you can't tell for instance
    // whether the pitch classes 0, 3, 6, 9 are C Dim 7,
    // D# Dim 7, F# Dim 7, or A Dim 7. In my Lisp
    // implementation, I use the surrounding musical context
    // (including pitch names derived from the combination of
    // MIDI and morphetic pitch numbers) to attempt to resolve
    // any ambiguities, but in this JavaScript implementation, I
    // just assume it's F# Dim 7 (or G Dim 7 or "G# Dim 7
    // respectively).
    "F# Dim 7", "G Dim 7", "G# Dim 7",
    "C half dim 7", "Db half dim 7", "D half dim 7", "Eb half dim 7", "E half dim 7", "F half dim 7",
    "F# half dim 7", "G half dim 7", "Ab half dim 7", "A half dim 7", "Bb half dim 7", "B half dim 7",
    "C dim", "Db dim", "D dim", "Eb dim", "E dim", "F dim",
    "F# dim", "G dim", "Ab dim", "A dim", "Bb dim", "B dim",
    "C minor 7", "Db minor 7", "D minor 7", "Eb minor 7", "E minor 7", "F minor 7",
    "F# minor 7", "G minor 7", "Ab minor 7", "A minor 7", "Bb minor 7", "B minor 7"
  ];

  function connect_or_not(singlescore, doublescore){
    // Tom Collins 26/10/2011.
    // In
    // singlescore Array mandatory
    // doublescore Integer mandatory
    // Out Boolean
    // This is how Pardo and Birmingham (2002) decide whether to unite two
    // previously separate segments.
    var connected = 0;
    if (doublescore >= singlescore[0] + singlescore[1]){
      connected = 1;
    }
    return connected;
  }

  function score_segment_against_template(segment, template){
    // Tom Collins 26/10/2011.
    // In
    // segment Object mandatory
    // template Array mandatory
    // Out Integer
    // This is Pardo and Birmingham's (2002) scoring function.

    var d = segment.points.length;
    var t = template.length;
    var member = new Array(d);
    for (let i = 0; i < d; i++){
      member[i] = segment.points[i][1] % 12;
    }
    var n = 0;
    var m = 0;
    var p = 0;
    for (let i=0; i<t; i++){
      if (member.indexOf(template[i]) === -1){
        m++;
      }
    }
    for (let k=0; k<d; k++){
      if (template.indexOf(member[k]) === -1){
        n++;
      }
      else {
        p++;
      }
    }
    return(p - m - n);
  }

  function find_segment_against_template(segment, template_set){
    // Tom Collins 26/10/2011.
    // In
    // segment Array mandatory
    // template_set Array mandatory
    // This function returns the chord template to which the input segment is
    // best matched, according to Pardo and Birmingham's (2002) scoring function.
    var d = template_set.length;
    var storage = new Array(d);
    for (let k = 0; k < d; k++){
      storage[k] = score_segment_against_template(segment, template_set[k]);
    }
    var mam = max_argmax(storage);
    var k = mam[1];
    return { "score": storage[k], "index": k, "vector": template_set[k] };
    //return(outtemplate);
  }

  function unique_rows(point_set){
    // Tom Collins 16/12/2014.
    // In
    // point_set Array mandatory
    // Out Array
    // The only argument to this function is an array consisting of numeric
    // arrays of uniform dimension (what I call a point set). This function
    // returns the unique elements in lexicographic order as first argument. As
    // second argument are the indices of each element from the input array.

    // Catch empty arrays.
    var n = point_set.length;
    if (n == 0){
      return [[], []];
    }

    // Sort the rows of a copy of the dataset.
    var E = sort_rows(point_set);
    var F = E[0];
    var g = E[1];
    // Create a new variable that will contain the unique rows of the dataset.
    var k = point_set[0].length;
    var U = new Array(n);
    // Create a new variable that will contain the indices of each unique row in
    // the original dataset.
    var v = new Array(n);
    U[0] = F[0];
    v[0] = [g[0]];
    var i = 1; // Increment over F and g.
    var j = 1; // Increment over U and v.
    while (i < n){
      if (array_equals(F[i],F[i - 1])){
        v[j - 1].push(g[i]);
      }
      else {
        U[j] = F[i];
        v[j] = [g[i]];
        j++;
      }
      i++;
    }
    return [U.slice(0, j), v.slice(0, j)];
  }

  /**
   * This function labels input sets of notes (in `segments`), with chord names as
   * provided in `templates` and `lookup`. Contiguous sets of notes may be
   * combined and given the same label, if it is parsimonious to do so according
   * to the algorithm. There is also some tolerance for non-chord tones. The
   * function is an implementation of the forwards HarmAn algorithm of Pardo and
   * Birmingham (2002).
   *
   * @author Tom Collins
   * @comment 26th October 2011
   * @tutorial chord-labelling-1
   * @param {Segment[]} segments - An array of segments
   * @param {number[][]} templates - An array of pitch-class sets.
   * @param {string[]} lookup - An array of strings paraellel to templates, giving
   * the interpretable label of each pitch-class set.
   * @return {Segment[]} An array of segments where the additional, extra
   * properties of name, index, and score have been populated, so as to (possibly
   * combine) and label the chords from the input `segments` with names from
   * `templates` and `lookup`.
   *
   * @example
   *     var ps = [
   *   [0, 45, 4], [0.5, 52, 3.5], [1, 59, 0.5], [1.5, 60, 2.5],
   *   [4, 41, 4], [4.5, 48, 3.5], [5, 55, 0.5], [5.5, 57, 2.5]
   * ];
   * var seg = segment(ps);
   * harman_forward(seg, chord_templates_pbmin7ths, chord_lookup_pbmin7ths);
   * â†’
   * [
   *   {
   *     "ontime": 0,
   *     "offtime": 4.5,
   *     "points": [
   *                 [0, 45, 51, 4, 0],
   *                 [0.5, 52, 55, 3.5, 0],
   *                 [1, 59, 59, 0.5, 0],
   *                 [1.5, 60, 60, 2.5, 0],
   *                 [4, 41, 49, 4, 0]
   *               ],
   *     "name": "A minor",
   *     "index": 33,
   *     "score": 6
   *   },
   *   {
   *     "ontime": 4.5,
   *     "offtime": 8,
   *     "points": [
   *                 [4, 41, 49, 4, 0],
   *                 [4.5, 48, 53, 3.5, 0],
   *                 [5, 55, 57, 0.5, 0],
   *                 [5.5, 57, 58, 2.5, 0]
   *               ],
   *     "name": "F major",
   *     "index": 5,
   *     "score": 6
   *   }
   * ]
   */
  function harman_forward(segments, templates, lookup){
    var L = segments.length;
    var lab = new Array();
    // Get the score for the very first segment.
    if (L > 0) {
      currentfind = find_segment_against_template(segments[0], templates);
      currentscore = currentfind.score;
    }
    // No testing to be done here, because there is no previous segment with
    // which to combine it, so just add it to lab.
    lab[0]= {
      "ontime": segments[0].ontime, "offtime": segments[0].offtime,
      "points": segments[0].points, "score": currentscore,
      "index": currentfind.index, "name": lookup[currentfind.index]
    };
    // console.log('lab[0].points:', lab[0].points);
    var i = 1; // Iterate over the remaining elements of segments.
    var j = 1; // Iterate over entries to lab.
    while (i < L){
      // Get the score from the last segment.
      var lastscore = lab[j - 1].score;
      // console.log('lastscore:', lastscore);
      // Get the score from the current segment.
      var currentfind = find_segment_against_template(segments[i], templates);
      var currentscore = currentfind.score;
      // console.log('currentscore:', currentscore);
      // Get the score from combining the last and current segments.
      // In my opinion (although I've not read the paper recently), the contents
      // of these segments ought to be de-duplicated as they are combined, e.g.
      // var duopoints = unique_rows(lab[j - 1].points.concat(segments[i].points));
      // This seems to cause over-segmentation, however, so I'm not doing this for
      // now. I am doing a final de-dupe before returning.
      var duopoints = [lab[j - 1].points.concat(segments[i].points)];
      var duosegments = {
        "ontime": lab[j - 1].ontime,
        "offtime": segments[i].offtime,
        "points": duopoints[0]
      };
      // console.log('duosegments.points:', duosegments.points);
      var combinedfind = find_segment_against_template(duosegments, templates);
      var combinedscore = combinedfind.score;
      // console.log('combinedscore:', combinedscore);

      // The test!
      if (combinedscore < lastscore + currentscore){
        // Label separately.
        lab.push({
          "ontime": segments[i].ontime, "offtime": segments[i].offtime,
          "points": segments[i].points, "score": currentscore,
          "index": currentfind.index, "name": lookup[currentfind.index]
        });
        j++;
      }
      else {
        // Label the combined segment.
        lab.pop();
        lab.push({
          "ontime": duosegments.ontime, "offtime": duosegments.offtime,
          "points": duosegments.points, "score": combinedscore,
          "index": combinedfind.index, "name": lookup[combinedfind.index]
        });
      }
      // console.log('lab[' + (j - 1) + '].points:', lab[j - 1].points);
      i++;
    }
    return lab.map(function(s){
      // De-dupe members of the points property.
      var psIdx = unique_rows(s.points);
      s.points = psIdx[0];
      return s;
    });
  }


  // Informal testing
  // import segment from './segment'
  // var chord_templates_pbmin7ths = [
  //   // Major triads
  //   [0, 4, 7], [1, 5 ,8], [2, 6, 9], [3, 7, 10], [4, 8, 11], [5 ,9, 0],
  //   [6, 10, 1], [7, 11, 2], [8, 0, 3], [9, 1, 4], [10, 2, 5], [11, 3, 6],
  //   // Dominant 7th triads
  //   [0, 4, 7, 10], [1, 5 ,8, 11], [2, 6, 9, 0], [3, 7, 10, 1],
  //   [4, 8, 11, 2], [5 ,9, 0, 3], [6, 10, 1, 4], [7, 11, 2, 5],
  //   [8, 0, 3, 6], [9, 1, 4, 7], [10, 2, 5 ,8], [11, 3, 6, 9],
  //   // Minor triads
  //   [0, 3, 7], [1, 4, 8], [2, 5 ,9], [3, 6, 10], [4, 7, 11], [5 ,8, 0],
  //   [6, 9, 1], [7, 10, 2], [8, 11, 3], [9, 0, 4], [10, 1, 5], [11, 2, 6],
  //   // Fully diminished 7th
  //   [0, 3, 6, 9], [1, 4, 7, 10], [2, 5 ,8, 11],
  //   // Half diminished 7th
  //   [0, 3, 6, 10], [1, 4, 7, 11], [2, 5 ,8, 0], [3, 6, 9, 1],
  //   [4, 7, 10, 2], [5 ,8, 11, 3], [6, 9, 0, 4], [7, 10, 1, 5],
  //   [8, 11, 2, 6], [9, 0, 3, 7], [10, 1, 4, 8], [11, 2, 5 ,9],
  //   // Diminished triad
  //   [0, 3, 6], [1, 4, 7], [2, 5 ,8], [3, 6, 9], [4, 7, 10], [5 ,8, 11],
  //   [6, 9, 0], [7, 10, 1], [8, 11, 2], [9, 0, 3], [10, 1, 4],
  //   [11, 2, 5],
  //   // Minor 7th
  //   [0, 3, 7, 10], [1, 4, 8, 11], [2, 5 ,9, 0], [3, 6, 10, 1],
  //   [4, 7, 11, 2], [5 ,8, 0, 3], [6, 9, 1, 4], [7, 10, 2, 5],
  //   [8, 11, 3, 6], [9, 0, 4, 7], [10, 1, 5 ,8], [11, 2, 6, 9]
  // ];
  //
  // var chord_lookup_pbmin7ths = [
  //   "C major", "Db major", "D major", "Eb major", "E major", "F major",
  //   "F# major", "G major", "Ab major", "A major", "Bb major", "B major",
  //   "C 7", "Db 7", "D 7", "Eb 7", "E 7", "F 7",
  //   "F# 7", "G 7", "Ab 7", "A 7", "Bb 7", "B 7",
  //   "C minor", "Db minor", "D minor", "Eb minor", "E minor", "F minor",
  //   "F# minor", "G minor", "Ab minor", "A minor", "Bb minor", "B minor",
  //   // Because Pardo & Birmingham (2002) only use MIDI note,
  //   // there is a bit of an issue with diminished 7th chords
  //   // (next three labels), as you can't tell for instance
  //   // whether the pitch classes 0, 3, 6, 9 are C Dim 7,
  //   // D# Dim 7, F# Dim 7, or A Dim 7. In my Lisp
  //   // implementation, I use the surrounding musical context
  //   // (including pitch names derived from the combination of
  //   // MIDI and morphetic pitch numbers) to attempt to resolve
  //   // any ambiguities, but in this JavaScript implementation, I
  //   // just assume it's F# Dim 7 (or G Dim 7 or "G# Dim 7
  //   // respectively).
  //   "F# Dim 7", "G Dim 7", "G# Dim 7",
  //   "C half dim 7", "Db half dim 7", "D half dim 7", "Eb half dim 7", "E half dim 7", "F half dim 7",
  //   "F# half dim 7", "G half dim 7", "Ab half dim 7", "A half dim 7", "Bb half dim 7", "B half dim 7",
  //   "C dim", "Db dim", "D dim", "Eb dim", "E dim", "F dim",
  //   "F# dim", "G dim", "Ab dim", "A dim", "Bb dim", "B dim",
  //   "C minor 7", "Db minor 7", "D minor 7", "Eb minor 7", "E minor 7", "F minor 7",
  //   "F# minor 7", "G minor 7", "Ab minor 7", "A minor 7", "Bb minor 7", "B minor 7"
  // ];

  // var ps = [
  //   [0, 45, 51, 4, 0], [0, 72, 67, 4, 1], [0, 76, 69, 4, 1], [0.5, 52, 55, 3.5, 0], [1, 59, 59, 0.5, 0], [1.5, 60, 60, 2.5, 0],
  //   [4, 41, 49, 4, 0], [4, 72, 67, 4, 1], [4, 77, 70, 4, 1], [4.5, 48, 53, 3.5, 0], [5, 55, 57, 0.5, 0], [5.5, 57, 58, 2.5, 0]
  // ];
  // var seg = segment(ps);
  // console.log('seg:', seg);
  // var lbl = harman_forward(
  //   seg,
  //   chord_templates_pbmin7ths,
  //   chord_lookup_pbmin7ths
  // );
  // console.log("lbl:", JSON.stringify(lbl));

  // var ps = [
  //   [-1, 74, 68, 0.75, 0], [-0.25, 71, 66, 0.25, 0],
  //   [0, 55, 57, 0.5, 1], [0, 74, 68, 1, 0], [0.5, 59, 59, 0.5, 1], [1, 62, 61, 0.5, 1], [1.5, 59, 59, 0.5, 1], [2, 62, 61, 0.5, 1], [2, 67, 64, 1, 0], [2.5, 59, 59, 0.5, 1],
  //   [3, 57, 58, 0.5, 1], [3, 66, 63, 1, 0], [3.5, 60, 60, 0.5, 1], [4, 62, 61, 0.5, 1], [4.5, 60, 60, 0.5, 1], [5, 62, 61, 0.5, 1], [5, 81, 72, 0.75, 0], [5.5, 60, 60, 0.5, 1], [5.75, 78, 70, 0.25, 0],
  //   [6, 54, 56, 0.5, 1], [6, 81, 72, 1, 0], [6.5, 57, 58, 0.5, 1], [7, 62, 61, 0.5, 1], [7.5, 57, 58, 0.5, 1], [8, 62, 61, 0.5, 1], [8, 72, 67, 1, 0], [8.5, 57, 58, 0.5, 1],
  //   [9, 55, 57, 0.5, 1], [9, 71, 66, 1, 0], [9.5, 59, 59, 0.5, 1], [10, 62, 61, 0.5, 1], [10.5, 59, 59, 0.5, 1]
  // ];
  // var seg = segment(ps);
  // var lbl = harman_forward(
  //   seg,
  //   chord_templates_pbmin7ths,
  //   chord_lookup_pbmin7ths
  // );
  // console.log('lbl:', JSON.stringify(lbl));

  function points_belonging_to_interval(point_set, a, b){
    // Tom Collins 25/10/2011.
    // In
    // point_set Array mandatory
    // a Number mandatory
    // b Number mandatory
    // Out Array
    // For a time interval [a, b), this function will return points from the
    // input point_set that sound during the time interval.

    var L = point_set.length;
    var segment = new Array(L);
    var i = 0;
    var j = 0;
    while (i < L) {
      if (point_set[i][0] < b && point_set[i][0] + point_set[i][3] > a) {
        segment[j] = point_set[i];
        j++;
      }
      if (point_set[i][0] >= b) {
        i = L;
      }
      i++;
    }
    return segment.slice(0, j);
  }

  function segment(
    pointSet, onAndOff = true, onIdx = 0, durIdx = 3
  ){
    // Tom Collins 25/10/2011.
    // In
    // pointSet Array mandatory
    // onAndOff Boolean optional
    // onIdx Integer optional
    // durIdx Integer optional
    // Out Array
    // This function will take a point set as input, calculate the unique ontimes
    // and offtimes, and return collections of notes that sound at each of the
    // unique times. It is a utility function, used by the HarmAn_forward
    // algorithm and various Markov models.
    //
    // The onAndOff variable is set to true by default, and will use both ontimes
    // and offtimes in the segmentation. If set to false, only ontimes will lead
    // to the creation of new segments, which end where the next ontime occurs. If
    // offtimes are noisier (e.g., in transcribed data), onAndOff = false is a
    // better option.

    // Get all the ontimes.
    var L = pointSet.length;
    var ontimes = new Array(L);
    for (let i = 0; i < L; i++){
      ontimes[i] = pointSet[i][onIdx];
    }
    // Even if onAndOff = false, we still need the maximal offtime for the ending
    // time of the final segment, so might as well create offtimes variable
    // anyway.
    var offtimes = new Array(L);
    for (let i = 0; i < L; i++){
      offtimes[i] = pointSet[i][onIdx] + pointSet[i][durIdx];
    }

    // Calculate the unique times.
    if (onAndOff){
      var segtimes = ontimes.concat(offtimes);
    }
    else {
      var segtimes = ontimes;
    }
    segtimes.sort(function(a, b){ return a - b });
    var uniquetimes = get_unique(segtimes);

    // For each unique time, find the notes that sound at this time.
    var d = uniquetimes.length;
    var segments = [];
    if (onAndOff){
      for (let k = 0; k < d - 1; k++){
        var a = uniquetimes[k];
        var b = uniquetimes[k + 1];
        // Test that this is really a segment, and not an artifact from rounding
        // tuplet on/offtimes.
        if (b - a > .00002){
          segments.push({
            "ontime": a,
            "offtime": b,
            "points": points_belonging_to_interval(pointSet, a, b)
          });
        }
      }
    }
    else {
      for (let k = 0; k < d; k++){
        var a = uniquetimes[k];
        if (k < d - 1){
          var b = uniquetimes[k + 1];
        }
        else {
          var b = max_argmax(offtimes)[0];
        }
        // Test that this is really a segment, and not an artifact from rounding
        // tuplet on/offtimes.
        if (b - a > .00002){
          segments.push({
            "ontime": a,
            "offtime": b,
            "points": points_belonging_to_interval(pointSet, a, b)
          });
        }
      }
    }
    return segments;
  }

  // Array operations.
  function append_array(an_array){
    // Tom Collins 23/12/2014.
    // In
    // an_array Array mandatory
    // Out Array
    // This function removes one level of brackets from an array.

    return an_array.reduce(function(a, b){
      return a.concat(b);
    }, []);

    // Old version.
    // var out_array = [];
    // for (let ia = 0; ia < an_array.length; ia++){
    //   for (let ib = 0; ib < an_array[ia].length; ib++){
    //     out_array.push(an_array[ia][ib]);
    //   }
    // }
  }

  function append_array_of_arrays(an_array){
    // Tom Collins 9/8/2015.
    // In
    // an_array Array mandatory
    // Out Array
    // In an array of arrays, this function identifies elements that are arrays
    // of arrays, as opposed to arrays whose first element is a string, and
    // removes one structural level from the former type of arrays.

    var out_array = [];
    for (let ia = 0; ia < an_array.length; ia++){
      if (typeof an_array[ia][0] == "string") {
        out_array.push(an_array[ia]);
      }
      else {
        for (let ib = 0; ib < an_array[ia].length; ib++){
          out_array.push(an_array[ia][ib]);
        }
      }

    }
    return out_array;

  }

  function array_object_index_of(myArray, searchTerm, property){
    // Joe on Stack Overflow 27/12/2014.
    // In
    // myArray Array mandatory
    // searchTerm Boolean, Number, or String mandatory
    // property String mandatory
    // Out Integer
    // In an array of objects, this function locates the index of the object
    // whose specifiable property is set to a specifiable value.
    // http://stackoverflow.com/questions/8668174/indexof-method-in-an-object-array

    for(var i = 0, len = myArray.length; i < len; i++){
      if (myArray[i][property] === searchTerm) return i;
    }
    return -1;
  }

  function array_object_index_of_array (myArray, searchArray, property){
    // Tom Collins 27/1/2015.
    // In
    // myArray Array mandatory
    // searchArray Array mandatory
    // property String mandatory
    // Out Integer
    // In an array of objects, this function locates the index of an array object
    // whose specifiable property is equal to a specifiable array.

    for(var i = 0, len = myArray.length; i < len; i++) {
      if (array_equals(myArray[i][property],searchArray)) return i;
    }
    return -1;
  }

  function array_sum(an_array){
    // Tom Collins 14/3/2015
    // In
    // an_array Array mandatory
    // Out Number
    // Returns the sum of elements of an array.

    return an_array.reduce(function(a, b){
      return a + b;
    }, 0);

    // Old version.
    // var count = 0;
    // for(var i = 0, n = an_array.length; i < n; i++){
    //  count += an_array[i];
    // }
    // return count;
  }

  function multiply_array_by_constant(an_array, a_constant){
    // Tom Collins 27/12/2014.
    // In
    // an_array Array mandatory
    // a_constant Number mandatory
    // Out Array
    // Two arguments are supplied to this function: an array and a constant. An
    // array is returned, containing the result of multiplying each element of
    // the input array by the constant.

    return an_array.map(function(a){ return a*a_constant; });

    // Old version.
    // var out_array = [];
    // for (let i = 0; i < an_array.length; i++){
    //   out_array.push(a_constant*an_array[i]);
    // }
    // return out_array;
  }

  function subtract_two_arrays(a, b){
    // Tom Collins 27/12/2014.
    // In
    // a Array mandatory
    // b Array mandatory
    // Out Array
    // Subtracts the second array from the first, element-by-element. It is
    // assumed that elements of array arguments are numbers, and the list
    // arguments are of the same length.

    var out_array = [];
    for (let i = 0; i < Math.min(a.length, b.length); i++){
      out_array.push(a[i] - b[i]);
    }
    return out_array;
  }

  /**
   * This function selects an element at random from the input array and returns
   * it. Optionally choices can be weighted by the parallel array cdf, which
   * contains a cumulative distribution function.
   *
   * @author Tom Collins
   * @comment 16th October 2014
   * @param {string|number|booelan[]} arr - An array.
   * @param {string|number|booelan[]} [cdf] - An array.
   * @return {(string|number|booelan)} An element from `arr`.
   *
   * @example
   *     choose_one(["jp", "mn", "hc"]);
   * â†’
   * "hc"
   */
  function choose_one(arr, cdf){
    if (arr.length > 0){
      if (cdf == undefined){
        var idx = Math.floor((Math.random()*arr.length));
        return arr[idx];
      }
      else {
        let jdx = 0;
        const rand = Math.random();
        while (rand >= cdf[jdx]){
          jdx++;
        }
        // console.log("jdx:", jdx)
        return arr[jdx]
      }
    }
  }

  function get_random_arbitrary(min, max){
    // Mozilla 11/2015.
    // In
    // min Number mandatory
    // max Number mandatory
    // Out Number
    // Returns a random number between min (inclusive) and max (exclusive).
    // From https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math/random

    return Math.random() * (max - min) + min;
  }

  function get_random_int(min, max){
    // Mozilla 11/2015.
    // In
    // min Integer mandatory
    // max Integer mandatory
    // Out Number
    // Returns a random integer between min (included) and max (excluded).
    // Using Math.round() will give you a non-uniform distribution!
    // From https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math/random

    return Math.floor(Math.random() * (max - min)) + min;
  }

  function get_random_int_inclusive(min, max){
    // Mozilla 11/2015.
    // In
    // min Integer mandatory
    // max Integer mandatory
    // Out Number
    // Returns a random integer between min (included) and max (included).
    // Using Math.round() will give you a non-uniform distribution!
    // From https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math/random

    return Math.floor(Math.random() * (max - min + 1)) + min;
  }

  function median(arr){
    // Tom Collins 28/8/2020.
    // In
    // arr Array mandatory
    // Out Number
    // This function returns the median of an input numeric array. No assumption
    // of sortedness. Tests input is all of type number, returning undefined if
    // this tests fails.

    if (arr.length === 0){
      return
    }
    const testAllNum = arr.every(function(x){ return typeof x === "number" });
    if (!testAllNum){
      console.log(
        "arr contains the wrong data types or is a mix of data types. " +
        "Returning undefined."
      );
      return
    }
    let sortedArr = arr.sort(function(x, y){
      return x - y
    });
    let n = sortedArr.length;
    if (n % 2 == 0){
      return (sortedArr[n/2 - 1] + sortedArr[n/2])/2
    }
    else {
      return sortedArr[(n - 1)/2]
    }
  }

  /**
   * This function calculates the sample standard deviation of an input array.
   *
   * @author Tom Collins
   * @comment 5th April 2020
   * @param {number|booelan[]} arr - An array.
   * @param {string|number|booelan[]} [cdf] - An array.
   * @return {(string|number|booelan)} An element from `arr`.
   *
   * @example
   *     std([727.7, 1086.5, 1091.0, 1361.3, 1490.5, 1956.1]);
   * â†’
   * 420.96
   */
  function std(x){
    const xbar = mean(x);
    let ss = 0;
    x.forEach(function(val){
      ss += Math.pow((val - xbar), 2);
    });
    return Math.sqrt(ss/(x.length - 1))
  }

  /**
   * This function calculates the sample skewness of an input array.
   *
   * @author Tom Collins
   * @comment 20th April 2020
   * @param {number[]} arr - An array.
   * @return {number} The calculated sample skewness.
   *
   * @example
   *     skewness([7.7, 10.5, 10.0, 13.3, 14.5, 19.1]);
   * â†’
   * 0.5251819
   */
  function skewness(x){
    const xbar = mean(x);
    let s3 = 0;
    x.forEach(function(val){
      s3 += Math.pow((val - xbar), 3);
    });
    let s2 = 0;
    x.forEach(function(val){
      s2 += Math.pow((val - xbar), 2);
    });
    if (s2 > 0){
      return (s3/x.length)/Math.pow(s2/x.length, 3/2)
    }
  }

  /**
   * This function calculates the median skewness (Pearson's second skewness
   * coefficient) of a sample contained in an input array.
   *
   * @author Tom Collins
   * @comment 3th September 2020
   * @param {number[]} arr - An array.
   * @return {number} The calculated median skewness.
   *
   * @example
   *     median_skewness([7.7, 10.5, 10.0, 13.3, 14.5, 19.1]);
   * â†’
   * 0.5019952
   */
  function median_skewness(x){
    const unbiasedVar = Math.pow(std(x), 2);
    const biasedVar = unbiasedVar*(x.length - 1)/x.length;
    return 3*(mean(x) - median(x))/Math.sqrt(biasedVar)
  }

  /**
   * This function calculates the sample excess kurtosis of an input array.
   *
   * @author Tom Collins
   * @comment 20th April 2020
   * @param {number[]} arr - An array.
   * @return {number} The calculated sample excess kurtosis.
   *
   * @example
   *     excess_kurtosis([7.7, 10.5, 10.0, 13.3, 14.5, 19.1]);
   * â†’
   * -0.7508978
   */
  function excess_kurtosis(x){
    const xbar = mean(x);
    let s4 = 0;
    x.forEach(function(val){
      s4 += Math.pow((val - xbar), 4);
    });
    let s2 = 0;
    x.forEach(function(val){
      s2 += Math.pow((val - xbar), 2);
    });
    if (s2 > 0){
      return (s4/x.length)/Math.pow(s2/x.length, 2) - 3
    }
  }

  function mod(x, n){
    // Tom Collins 28/8/2020.
    // In
    // x Number mandatory
    // n Positive integer mandatory
    // Out Number
    // This function returns x mod n. JavaScript's % operator returns remainder,
    // which can be negative. This function always returns non-negative values.
    // No checks are made on n being a positive integer.

    return x - (n*Math.floor(x/n))
  }

  /**
   * This function returns a random sample of elements from an input array, where
   * the sampling is without replacement.
   *
   * @author Tom Collins
   * @comment 30th June 2019
   * @param {Array<number|string|boolean>} arr - Input array
   * @param {number} sampleSize - Size of requested sample
   * @return {Array<number|string|boolean>} Output array
   *
   * @example
   *     sample_without_replacement(["a", "b", "c"], 2);
   * â†’
   * ["c", "a"]
   */
  function sample_without_replacement(arr, sampleSize){
    var n = arr.length;
    if (n > 0){
      if (n >= sampleSize){
        var idxUnchosen = new Array(n);
        for (var i = 0; i < n; i++){
          idxUnchosen[i] = i;
        }
        // console.log("idxUnchosen:", idxUnchosen);
        var idxChosen = [];
        while (idxChosen.length < sampleSize){
          // Choose one of the idxUnchosen.
          var idxChoice = get_random_int(0, idxUnchosen.length);
          // console.log("idxChoice:", idxChoice);
          // Add it to idxChosen.
          idxChosen.push(idxUnchosen[idxChoice]);
          // console.log("idxChosen:", idxChosen);
          // Remove it from idxUnchosen.
          idxUnchosen.splice(idxChoice, 1);
          // console.log("idxUnchosen:", idxUnchosen);
        }
        return idxChosen.map(function(idx){
          return arr[idx];
        });
      }
    }
  }

  // Interpolation.
  function index_1st_element_gt(item, an_array){
    // Tom Collins 27/12/2014.
    // In
    // item Number mandatory
    // an_array Array mandatory
    // Out Integer
    // This function takes two arguments: a real number x and an array L of real
    // numbers. It returns the index of the first element of L that is greater
    // than x.

    var idx = 0;
    var jdx = undefined;
    while (idx < an_array.length){
      if (an_array[idx] > item){
        jdx = idx;
        idx = an_array.length;
      }
      idx++;
    }
    return jdx;
  }

  // Projection.
  function lex_less_or_equal_triple(dj, v, di, k){
    // Tom Collins 16/12/2014.
    // In
    // dj Array mandatory
    // v Array mandatory
    // di Array mandatory
    // k Integer optional
    // This function is useful when testing whether dj + v is lexicographically
    // less than or equal to di. It is faster to check each element of dj + v and
    // di in turn, rather than calculate dj + v first.
    //
    // The function returns 1 if dj + v is 'less than' di, where 'less than' is
    // the lexicographic ordering. It returns -1 if dj + v is 'greater than' di,
    // and it returns 0 if dj + v equals di.
    //
    // In general, for two vectors u and w, this function finds the first index i
    // such that u(i) is not equal to w(i). If u(i) is less than w(i), then u is
    // 'less than' w. If w(i) is less than u(i), then w is 'less than' u. The other
    // possible outcome is that u equals w.

    if (k == undefined){
      k = dj.length;
    }
    // Logical outcome.
    var tf = 0;
    // Dimension of vector.
    var s = 0; // Increment over dj and v.
    var e; // Each element of E = dj + v;
    while (s < k){
      e = dj[s] + v[s];
      if (e > di[s]){
        tf = -1;
        s = k;
      }
      else {
        if (e == di[s]){
          s++;
        }
        else {
          tf = 1;
          s = k;
        }
      }
    }
    return tf;
  }

  function maximal_translatable_pattern(v, D, k, n){
    // Tom Collins 16/12/2014.
    // In
    // v Array mandatory
    // D Array mandatory
    // k Integer mandatory
    // n Integer mandatory
    // Out Array
    // This function calculates the maximal translatable pattern (MTP, Meredith
    // Lemstrom, & Wiggins, 2002) of the vector v in the point set D (containing
    // n k-dimensional points). The MTP P and indices I of datapoints forming the
    // MTP are returned. It is assumed that the point set D is in lexicograhic
    // order.

    var P = new Array(n);
    var I = new Array(n);
    var i = 0; // Increment over D.
    var j = 0; // Increment over E (= to D + v).
    var L = 0; // Increment over P.
    var tf; // Outcome of call to function lexLessOrEqualTriple.
    while (i < n){
      tf = lex_less_or_equal_triple(D[j], v, D[i], k);
      if (tf == -1){
        i++;
      }
      else {
        if (tf == 0){
          P[L] = D[j];
          I[L] = j;
          i++;
          j++;
          L++;
        }
        else {
          j++;
        }
      }
    }
    return [P.slice(0, L), I.slice(0, L)];
  }

  // import intersect from './intersect'

  /**
   * This function calculates the difference between each pair of points in P and Q, sorts
   % by frequency of occurrence, and then returns the frequency of the most
   % frequently occurring difference vector, divided by the maximum of the
   % number of points in P and Q. If P is a translation of Q, then the
   % cardinality score is 1; if no two pairs of P points and Q points are
   % translations, then the cardinality score is zero; otherwise it is
   % somewhere between the two.
   *
   * @author Tom Collins
   * @comment 4th February 2020
   * @param {PointSet} P - A point set
   * @param {PointSet} Q - A point set
   * @return {number} Output decimal and array
   *
   * @example
   *     cardinality_score([[1, 1], [1, 3], [1, 4], [2, 2], [3, 1], [4, 1], [4, 4]], [[3, 4], [3, 6], [3, 7], [4, 2], [5, 4], [5, 5], [6, 7], [7, 1]])
   * â†’
   * [0.625, [2, 3]]
   */
  function cardinality_score(P, Q, allowTrans = true){
    const m = P.length;
    const n = Q.length;
    let numerator, maxTransVec;
    if (allowTrans){
      // Calculate the difference array, but leave it as a vector.
      const k = P[0].length;
      const bigN = m*n;
      let bigV = new Array(bigN);
      let bigL = 0; // Increment to populate V.
      for (let i = 0; i < m; i++){
        for (let j = 0; j < n; j++){
          bigV[bigL] = subtract_two_arrays(Q[j], P[i]);
          bigL++;
        }
      }
      const bigV2 = count_rows(bigV);
      const ma = max_argmax(bigV2[1]);
      numerator = ma[0];
      maxTransVec = bigV2[0][ma[1]];
    }
    else {
      console.log("YOU NEED TO WRITE THE INTERSECT FUNCTION!");
      return
    }
    // console.log("numerator:", numerator)
    // console.log("maxTransVec:", maxTransVec)
    let sCard = numerator/Math.max(m, n);

    return [sCard, maxTransVec]

  }

  // String operations.
  function locations(substring, string){
    // Tom Collins 18/2/2016.
    // In
    // substring String mandatory
    // string String mandatory
    // This function is from vcsjones on stackoverflow, for finding the indices
    // of multiple occurrences of a substring in a string. I thought it would be
    // possible to call str.search(e) where e is a regexp with global modifier,
    // but this did not seem to work.
    // http://stackoverflow.com/questions/10710345/finding-all-indexes-of-a-specified-character-within-a-string

    var a=[],i=-1;
    while((i=string.indexOf(substring,i+1)) >= 0) a.push(i);
    return a;
  }

  function my_last_string(a_string){
    // Tom Collins 20/9/2015.
    // In
    // a_string String mandatory
    // Out String
    // This function returns the last element of a string as a string.

    if (a_string.length === 0){
      return "";
    }
    else {
      return a_string[a_string.length - 1];
    }
  }

  function string_separated_string2array(substring, a_string){
    // Tom Collins 9/8/2015.
    // In
    // substring String mandatory
    // a_string String mandatory
    // Out Array
    // This function converts a string (second argument) interspersed with
    // occurrences of a substring (first argument) into an array, where each
    // element is a string preceding or proceeding the substring.

    var an_array = a_string.split(substring);
    for (let i = 0; i < an_array.length; i++){
      an_array[i] = an_array[i].trim();
    }
    return an_array;
  }

  /**
   * This function returns the Farey sequence of order *n*.
   *
   * @author Tom Collins
   * @comment 21st January 2018
   * @param {number} n - Order of the Farey sequence
   * @return {number[]} Farey sequence, deduplicated and in ascending order
   *
   * @example
   *     farey(6);
   * â†’
   * [0, 0.16667, 0.25, 0.33333, 0.5, 0.66667, 0.75, 0.83333, 1]
   */
  function farey$1(n){
    var out_arr = [];
    for (let m = 2; m <= n; m++){
      var fracs = [];
      for (let i = 1; i < m; i++){
        // Round to 5 d.p. and push to fracs.
        fracs.push(Math.round(100000*i/m)/100000);
      }
      out_arr = out_arr.concat(fracs);
    }
    // Sort and get unique elements.
    out_arr = get_unique(out_arr.sort(function(a, b){return a - b;}));
    // Stick 0 at the front and 1 at the end.
    out_arr.unshift(0);
    out_arr.push(1);
    return out_arr;
  }

  /**
   * This function quantises time values in the input point set `D` by mapping
   * their fractional parts to the closest member of the Farey sequence of order
   * *n* (second argument). In a standard point set, time values are located in
   * the first (ontime) and third (duration) indices of each point, hence the
   * default argument for `dim` is `[0, 3]`.
   *
   * @author Tom Collins
   * @comment 21st January 2018
   * @param {PointSet} D - A point set
   * @param {number[]} [Fn] - Usually a Farey sequence
   * @param {number[]} [dim] - An array of nonnegative integers indicating the
   * indices of time values in the point set.
   * @return {PointSet} Quantised point set
   *
   * @example
   *     var ps = [
   *   [1.523, 60, 0.980],
   *   [2.873, 72, 0.192]
   * ];
   * var fareySeq = [0, 0.5, 1];
   * var dimArr = [0, 2];
   * farey_quantise(ps, fareySeq, dimArr);
   * â†’
   * [
   *   [1.5, 60, 1],
   *   [3, 72, 0.5]
   * ];
   */
  function farey_quantise(D, Fn, dim){
    if (Fn === undefined){
      Fn = farey(4);
    }
    if (dim === undefined){
      dim = [0, 3];
    }
    D.forEach(function(d){
      dim.map(function(j){
        // Compute the difference between each time value and Farey sequence
        // member. The fractional part is d[j] - Math.floor(d[j]).
        var diffs = Fn.map(function(x){
          return Math.abs(d[j] - Math.floor(d[j]) - x);
        });
        // Compute minimum difference.
        var ma = min_argmin(diffs);
        // Reset the fractional part of the time value to be the Farey sequence
        // member corresponding to the minimum difference.
        d[j] = Math.floor(d[j]) + Fn[ma[1]];
        // Can't have zero durations so correct any of these.
        if ((j == 2 || j == 3) && d[j] == 0){
          d[j] = Fn[1];
        }
        return;
      });
      return d;
    });
    return D;
  }

  /**
   * This function will return a string referring to a parameter value in the page
   * URL, given the parameter name as its only argument.
   *
   * @author Tom Collins
   * @comment 13th April 2020
   * @param {string} name - Referring to a parameter name in the page URL.
   * @return {string} Referring to the corresponding parameter value in the page
   * URL.
   *
   * @example
   *     Assuming a URL of https://something.com/index.html?r=0,3,0&c=0,2,3
   *     get_parameter_by_name("r")
   * â†’
   * "0,3,0"
   */
  function get_parameter_by_name(name){
    let match = RegExp('[?&]' + name + '=([^&]*)').exec(window.location.search);
    return match && decodeURIComponent(match[1].replace(/\+/g, ' '))
  }

  /**
   * This function calculates the sample standard deviation of an input array.
   *
   * @author Tom Collins
   * @comment 13th April 2020
   * @param {number|booelan[]} arr - An array.
   * @param {string|number|booelan[]} [cdf] - An array.
   * @return {(string|number|booelan)} An element from `arr`.
   *
   * @example
   *     copy_to_clipboard("Hi there!")
   * â†’
   * (Copies the text "Hi there!" to the clipboard.)
   */
  function copy_to_clipboard(copyText, alertText){
    let dummy = document.createElement("input");
    document.body.appendChild(dummy);
    dummy.setAttribute("id", "dummy_id");
    document.getElementById("dummy_id").value = copyText;
    dummy.select();
    dummy.setSelectionRange(0, 99999);
    document.execCommand("copy");
    document.body.removeChild(dummy);
    if (alertText == undefined){
      alert("Link copied to clipboard:\n" + copyText);
    }
    else {
      alert(alertText);
    }
  }

  /**
   * @file Welcome to the API for MAIA Util!
   *
   * MAIA Util is a JavaScript package used by Music Artificial Intelligence
   * Algorithms, Inc. in various applications that we have produced or are
   * developing currently.
   *
   * If you already know about JavaScript app development and music computing,
   * then probably the best starting point is the
   * [NPM package](https://npmjs.com/package/maia-util/).
   *
   * If you have a music computing background but know little about JavaScript,
   * then the tutorials menu is a good place to start. There are also some
   * fancier-looking demos available
   * [here](http://tomcollinsresearch.net/mc/ex/),
   * all of which involve MAIA Util methods to some degree.
   *
   * If you don't know much about music or music computing, then the
   * [fancier-looking demos](http://tomcollinsresearch.net/mc/ex/) are still the
   * best place to start, to get hooked on exploring web-based, interactive music
   * interfaces.
   *
   * This documentation is in the process of being completed. Some functions have
   * not had their existing documentation converted to JSDoc format yet.
   *
   * @version 0.2.20
   * @author Tom Collins and Christian Coulon
   * @copyright 2016-2020
   *
   */


  const append_ontimes_to_time_signatures$1 = append_ontimes_to_time_signatures;
  const bar_and_beat_number_of_ontime$1 = bar_and_beat_number_of_ontime;
  const clef_sign_and_line2clef_name$1 = clef_sign_and_line2clef_name;
  const convert_1st_bar2anacrusis_val$1 = convert_1st_bar2anacrusis_val;
  const default_page_and_system_breaks$1 = default_page_and_system_breaks;
  const group_grace_by_contiguous_id$1 = group_grace_by_contiguous_id;
  const guess_morphetic$1 = guess_morphetic;
  const midi_note_morphetic_pair2pitch_and_octave$1 = midi_note_morphetic_pair2pitch_and_octave;
  const mnn2pitch_simple$1 = mnn2pitch_simple;
  const guess_morphetic_in_C_major = guess_morphetic_in_c_major;
  const guess_morphetic_in_c_major$1 = guess_morphetic_in_c_major;
  const nos_symbols_and_mode2key_name$1 = nos_symbols_and_mode2key_name;
  const ontime_of_bar_and_beat_number$1 = ontime_of_bar_and_beat_number;
  const pitch_and_octave2midi_note_morphetic_pair$1 = pitch_and_octave2midi_note_morphetic_pair;
  const remove_duplicate_clef_changes$1 = remove_duplicate_clef_changes;
  const resolve_expressions$1 = resolve_expressions;
  const row_of_max_ontime_leq_ontime_arg$1 = row_of_max_ontime_leq_ontime_arg;
  const row_of_max_bar_leq_bar_arg$1 = row_of_max_bar_leq_bar_arg;
  const sort_points_asc$1 = sort_points_asc;
  const sort_points_asc_by_id$2 = sort_points_asc_by_id$1;
  const staff_voice_xml2staff_voice_json$1 = staff_voice_xml2staff_voice_json;
  const comp_obj2note_point_set$1 = comp_obj2note_point_set;
  const split_point_set_by_staff$1 = split_point_set_by_staff;
  const copy_array_object$1 = copy_array_object;
  const count_rows$1 = count_rows;
  const tonic_pitch_closest$1 = tonic_pitch_closest;
  const fifth_steps_mode$1 = fifth_steps_mode;
  const aarden_key_profiles$1 = aarden_key_profiles;
  const krumhansl_and_kessler_key_profiles$1 = krumhansl_and_kessler_key_profiles;
  const chord_templates_pbmin7ths$1 = chord_templates_pbmin7ths;
  const chord_lookup_pbmin7ths$1 = chord_lookup_pbmin7ths;
  const connect_or_not$1 = connect_or_not;
  const find_segment_against_template$1 = find_segment_against_template;
  const HarmAn_forward = harman_forward;
  const harman_forward$1 = harman_forward;
  const points_belonging_to_interval$1 = points_belonging_to_interval;
  const score_segment_against_template$1 = score_segment_against_template;
  const segment$1 = segment;
  const append_array$1 = append_array;
  const append_array_of_arrays$1 = append_array_of_arrays;
  const array_object_index_of$1 = array_object_index_of;
  const array_object_index_of_array$1 = array_object_index_of_array;
  const array_sum$1 = array_sum;
  const cyclically_permute_array_by$1 = cyclically_permute_array_by;
  const max_argmax$1 = max_argmax;
  const min_argmin$1 = min_argmin;
  const multiply_array_by_constant$1 = multiply_array_by_constant;
  const copy_point_set$1 = copy_point_set;
  const get_unique$1 = get_unique;
  const index_point_set$1 = index_point_set;
  const lex_more$1 = lex_more;
  const sort_rows$1 = sort_rows;
  const subtract_two_arrays$1 = subtract_two_arrays;
  const restrict_point_set_in_nth_to_xs$1 = restrict_point_set_in_nth_to_xs;
  const unique_rows$1 = unique_rows;
  const choose_one$1 = choose_one;
  const corr$1 = corr;
  const get_random_arbitrary$1 = get_random_arbitrary;
  const get_random_int$1 = get_random_int;
  const get_random_int_inclusive$1 = get_random_int_inclusive;
  const mean$1 = mean;
  const median$1 = median;
  const std$1 = std;
  const skewness$1 = skewness;
  const median_skewness$1 = median_skewness;
  const excess_kurtosis$1 = excess_kurtosis;
  const mod$1 = mod;
  const sample_without_replacement$1 = sample_without_replacement;
  const index_1st_element_gt$1 = index_1st_element_gt;
  const lex_less_or_equal_triple$1 = lex_less_or_equal_triple;
  const orthogonal_projection_not_unique_equalp$1 = orthogonal_projection_not_unique_equalp;
  const maximal_translatable_pattern$1 = maximal_translatable_pattern;
  const cardinality_score$1 = cardinality_score;
  const locations$1 = locations;
  const my_last_string$1 = my_last_string;
  const string_separated_string2array$1 = string_separated_string2array;
  const farey$2 = farey$1;
  const farey_quantise$1 = farey_quantise;
  const get_parameter_by_name$1 = get_parameter_by_name;
  const copy_to_clipboard$1 = copy_to_clipboard;

  var maiaUtil = {
    append_ontimes_to_time_signatures: append_ontimes_to_time_signatures$1,
    bar_and_beat_number_of_ontime: bar_and_beat_number_of_ontime$1,
    clef_sign_and_line2clef_name: clef_sign_and_line2clef_name$1,
    convert_1st_bar2anacrusis_val: convert_1st_bar2anacrusis_val$1,
    default_page_and_system_breaks: default_page_and_system_breaks$1,
    group_grace_by_contiguous_id: group_grace_by_contiguous_id$1,
    guess_morphetic: guess_morphetic$1,
    midi_note_morphetic_pair2pitch_and_octave: midi_note_morphetic_pair2pitch_and_octave$1,
    mnn2pitch_simple: mnn2pitch_simple$1,
    guess_morphetic_in_c_major: guess_morphetic_in_c_major$1,
    guess_morphetic_in_C_major,
    nos_symbols_and_mode2key_name: nos_symbols_and_mode2key_name$1,
    ontime_of_bar_and_beat_number: ontime_of_bar_and_beat_number$1,
    pitch_and_octave2midi_note_morphetic_pair: pitch_and_octave2midi_note_morphetic_pair$1,
    remove_duplicate_clef_changes: remove_duplicate_clef_changes$1,
    resolve_expressions: resolve_expressions$1,
    row_of_max_ontime_leq_ontime_arg: row_of_max_ontime_leq_ontime_arg$1,
    row_of_max_bar_leq_bar_arg: row_of_max_bar_leq_bar_arg$1,
    sort_points_asc: sort_points_asc$1,
    sort_points_asc_by_id: sort_points_asc_by_id$2,
    staff_voice_xml2staff_voice_json: staff_voice_xml2staff_voice_json$1,
    comp_obj2note_point_set: comp_obj2note_point_set$1,
    split_point_set_by_staff: split_point_set_by_staff$1,
    copy_array_object: copy_array_object$1,
    count_rows: count_rows$1,
    tonic_pitch_closest: tonic_pitch_closest$1,
    fifth_steps_mode: fifth_steps_mode$1,
    aarden_key_profiles: aarden_key_profiles$1,
    krumhansl_and_kessler_key_profiles: krumhansl_and_kessler_key_profiles$1,
    chord_templates_pbmin7ths: chord_templates_pbmin7ths$1,
    chord_lookup_pbmin7ths: chord_lookup_pbmin7ths$1,
    connect_or_not: connect_or_not$1,
    find_segment_against_template: find_segment_against_template$1,
    harman_forward: harman_forward$1,
    HarmAn_forward,
    points_belonging_to_interval: points_belonging_to_interval$1,
    score_segment_against_template: score_segment_against_template$1,
    segment: segment$1,
    append_array: append_array$1,
    append_array_of_arrays: append_array_of_arrays$1,
    array_object_index_of: array_object_index_of$1,
    array_object_index_of_array: array_object_index_of_array$1,
    array_sum: array_sum$1,
    cyclically_permute_array_by: cyclically_permute_array_by$1,
    max_argmax: max_argmax$1,
    min_argmin: min_argmin$1,
    multiply_array_by_constant: multiply_array_by_constant$1,
    copy_point_set: copy_point_set$1,
    get_unique: get_unique$1,
    index_point_set: index_point_set$1,
    lex_more: lex_more$1,
    sort_rows: sort_rows$1,
    subtract_two_arrays: subtract_two_arrays$1,
    restrict_point_set_in_nth_to_xs: restrict_point_set_in_nth_to_xs$1,
    unique_rows: unique_rows$1,
    choose_one: choose_one$1,
    corr: corr$1,
    get_random_arbitrary: get_random_arbitrary$1,
    get_random_int: get_random_int$1,
    get_random_int_inclusive: get_random_int_inclusive$1,
    mean: mean$1,
    median: median$1,
    std: std$1,
    skewness: skewness$1,
    median_skewness: median_skewness$1,
    excess_kurtosis: excess_kurtosis$1,
    mod: mod$1,
    sample_without_replacement: sample_without_replacement$1,
    index_1st_element_gt: index_1st_element_gt$1,
    lex_less_or_equal_triple: lex_less_or_equal_triple$1,
    orthogonal_projection_not_unique_equalp: orthogonal_projection_not_unique_equalp$1,
    maximal_translatable_pattern: maximal_translatable_pattern$1,
    cardinality_score: cardinality_score$1,
    locations: locations$1,
    my_last_string: my_last_string$1,
    string_separated_string2array: string_separated_string2array$1,
    farey: farey$2,
    farey_quantise: farey_quantise$1,
    get_parameter_by_name: get_parameter_by_name$1,
    copy_to_clipboard: copy_to_clipboard$1
  };

  return maiaUtil;

}());