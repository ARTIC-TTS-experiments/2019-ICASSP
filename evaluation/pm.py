""" Support for processing pitch-mark file
"""

import bisect
import pickle
import sys
import math


##
#  Class for representation of one particular pitch-mark. A pitch-mark is defined by its time instant and type.
#
class OnePm(object):

    ## Type of voiced pitch-mark, get by OnePm.type when OnePm.V == <code>True</code>
    type_V = 'V'
    ## Type of unvoiced pitch-mark, get by OnePm.type when OnePm.U == <code>True</code>
    type_U = 'U'
    ## Type of transitional pitch-mark, get by OnePm.type when OnePm.T == <code>True</code>
    type_T = 'T'

    ##
    #  Constructor.
    #
    #  @param time time instant of the pitch-mark [sec]
    #  @param type type of the pitch-mark
    #
    def __init__(self, time, type):
        self.__time = time
        self.__type = type

    ##
    # Property getting the time instant of the pitch-mark.
    #
    @property
    def time(self):
        return self.__time
    ##
    #  Property getting the type of the pitch-mark.
    #
    @property
    def type(self):
        return self.__type

    ##
    #  Property getting <code>True</code> when the pitch-mark type is voiced.
    #
    @property
    def V(self):
        return self.__type == OnePm.type_V
    ##
    #  Property getting <code>True</code> when the pitch-mark type is unvoiced.
    #
    @property
    def U(self):
        return self.__type == OnePm.type_U
    ##
    #  Property getting <code>True</code> when the pitch-mark type is transitional.
    #
    @property
    def T(self):
        return self.__type == OnePm.type_T

    ##
    # The pitch-mark <i>less-then</i> comparison method. It uses the time of pitch-mark
    # @param obj the object to compare the pitch-mark to
    # @param self
    #
    def __lt__(self, obj):
        return self.time < (obj.time if isinstance(obj, OnePm) else obj)
    ##
    # The pitch-mark <i>greater-then</i> comparison method. It uses the time of pitch-mark
    # @param obj the object to compare the pitch-mark to
    # @param self
    #
    def __gt__(self, obj):
        return self.time > (obj.time if isinstance(obj, OnePm) else obj)
    ##
    # The pitch-mark <i>equal-to</i> comparison method. It uses both the the time and type
    # of pitch-mark when compared to other OnePm object, or just time when compared to a
    # float value.
    #
    # @param obj the object to compare the pitch-mark to
    # @param self
    #
    def __eq__(self, obj):
        # Originally, `-` operator was used and worked OK but then the following error appeared:
        # TypeError: numpy boolean subtract, the `-` operator, is deprecated, use the bitwise_xor,
        # the `^` operator, or the logical_xor function instead.
        # => So, `^` was used instead
        # (another possibility is to use math.isclose() or numpy.isclose() functions)
        if   isinstance(obj, OnePm) :
             return self.type == obj.type and ((self.time > obj.time) ^ (self.time < obj.time)) == 0
        elif isinstance(obj, float) :
             return                           ((self.time > obj)      ^ (self.time < obj))      == 0
        else :
             return False

    #
    def is_close(self, obj, tol=0.0005):
        if isinstance(obj, OnePm):
            return self.type == obj.type and math.isclose(self.time, obj.time, abs_tol=tol)
        elif isinstance(obj, float):
            return math.isclose(self.time, obj, abs_tol=tol)
        else:
            return False

    ##
    # The method printing user-readable information about the pitch-mark object.
    # @param  self
    #
    def __str__(self):
        return u'pitch-mark({:.5f},{})'.format(self.time, self.type)
    ##
    # The method printing user-readable information about the pitch-mark object.
    # @param  self
    #
    def __repr__(self):
        return  'pitch-mark({:.5f},{})'.format(self.time, self.type)
    ##
    # Make the type hasheable. The hash value is equal to the hash value of the pitch-mark
    # time.
    #
    def __hash__(self) :
        return hash(self.__time)


    # constants - do not change!
    # @deprecated
    type_voiced         = type_V
    type_unvoiced       = type_U
    type_transitional   = type_T
    type_unsure         = '?'

    ##
    #  The method getting the time instant of the pitch-mark.
    #
    def get_time(self):
        return self.__time
    ##
    #  The method getting the type of the pitch-mark.
    #
    def get_type(self):
        return self.__type

    ##
    #  Test whether the pitch-mark type is voiced.
    #
    def is_voiced(self):
        return self.V
    ##
    #  Test whether the pitch-mark type is unvoiced.
    #
    def is_unvoiced(self):
        return self.U
    ##
    #  Test whether the pitch-mark type is transitional.
    #
    def is_transitional(self):
        return self.T

    # @deprecated



# ==============================


##
#  An auxiliary class. When F<sub>0</sub> is calculated by class Pm, boundary correction (alignment to pitch-mark)
#  is performed. Some functions return instance of this class which contains corrected (shifted) boundaries
#  and resulting F<sub>0</sub>.
#
class Segment(object):

    ##
    #  Constructor.
    #
    #  @param time1 begin of the segment (float)
    #  @param time2 end of the segment (float)
    #  @param f0 average F<sub>0</sub> between <i>time1</i> and <i>time2</i>
    #
    def __init__(self, time1, time2, f0):
        self.__time1 = time1
        self.__time2 = time2
        self.__f0 = f0


    ##
    #  Return the begin of the segment.
    #
    def get_time1(self):
        return self.__time1


    ##
    #  Return the end of the segment.
    #
    def get_time2(self):
        return self.__time2


    ##
    #  Return the average F<sub>0</sub> within the segment.
    #
    def get_f0(self):
        return self.__f0


# ====================


##
#  Class for manipulating sequences of pitch-marks corresponding to an utterance.
#
#  When seek the index for a time instance (boundary between 2 units) or calculate F<sub>0</sub> of an unit,
#  boundaries of this unit can be shifted to an surounding pitch-mark
#  according to the pre-set shift type (set_shift_type(), get_shift_type())
#  <ul>
#  <li><i> shift_none </i>    no shift is performed
#  <li><i> shift_left </i>    shift to the nearest pitch-mark on the left
#  <li><i> shift_right </i>   shift to the nearest pitch-mark on the right
#  <li><i> shift_nearest </i> shift to the nearest pitch-mark
#  </ul>
#  After this basic shift, an additional boundary correction between voiced and unvoiced unit can be performed,
#  first or last unvoiced pitch-mark is sought. It is activated by when following constants are used
#  <ul>
#  <li><i> shift_Left </i>
#  <li><i> shift_Right </i>
#  <li><i> shift_Nearest </i>
#  </ul>
#
#  When the nearest pitch-mark is sought (left, rigth or both), first the absolutely nearest pitch-mark is found,
#  if its in the area around the given time, then this pitch-mark is selected (no matter whether left or right
#  parameter was used). This area is defined using the variable <i> match_tol </i> (default value is 0.001 [sec]).
#
#  During the additional boundary correction, first/last unvoiced pitch mark is sought in the area given by
#  variable <i> uv_boundary_tol </i> (default value is 0.02 [sec]).
#

class Pm(object):


    # common class constants and variables
    shift_none    = 0
    shift_left    = 1
    shift_right   = 2
    shift_nearest = 3
    shift_Left    = shift_left
    shift_Right   = shift_right
    shift_Nearest = shift_nearest

    # boundary type
    bound_time = 'T'    # time (need not correspond to time of a concrete pitch-mark)
    bound_idx  = 'I'    # index of pitch-mark

    # some parametres used within calculating f0
    match_tol       = 0.001  # tolerance for pitch-mark placement
    uv_boundary_tol = 0.02   # tolerance for pitch-mark placement
                             # in case of boundary between voiced and unvoiced unit
                             # (first or last unvoiced pitch-mark is searched)
    voiced_ratio    = 0.5    # what minimum part of unit should be voiced as to the whole unit is voiced
    voiced_pm_dist  = 0.02   # maximum distance between 2 pitch-marks in the voiced signal
    invalid_f0      = -1     # f0 value for unvoiced units
    idx_continue    = -1     # if the 1st parameter (time) in get_f0 or get_segment is of this value,
                             # the corresponding 1st index will automatically take the same value
                             # as the previous end index


    ##
    #  Constructor.
    #
    #  @param filename name of text file containing pitch-marks
    #  @param shift_type
    #  @param times list of pitch-mark times
    #  @param types list of pitch-mark types
    #
    def __init__(self, filename = None, shift_type = shift_nearest, times=None, types=None):
        self.__fname = filename
        self.__last_idx = 0
        self.__pmlist = []
        self.__shift_type = shift_type

        # Read pitch-marks from file
        if filename is not None:
            self.read_file(filename)

        # Set up pitch-marks from times (and types)
        if times is not None and types is not None:
            if len(times) != len(types):
                raise RuntimeError('Pitch-mark times and types do not match!')
            self.set_pmks([OnePm(tm, tp) for tm, tp in zip(times, types)])
        elif times is not None:
            self.set_pmks([OnePm(t, OnePm.type_V) for t in times])

    ##
    #  Read a text file containing pitch-marks.
    #
    #  @param filename name of text file containing pitch-marks, or a file-like object from which to read the data.
    #
    def read_file(self, filename):
        try:

            # fname is an object from which to read the data
            if not isinstance(filename, str):
               self.__fname = 'Unknown source represented by %s' % str(filename)
               buffer = filename.readlines()
            # Read from "classic" file
            else:
               self.__fname = filename
               file = open(filename, 'r')
               buffer = file.readlines()
               file.close()
            # Process the buffer
            self.__process_file(buffer)
        except IOError:
            raise Exception('File "{}" cannot be processed...'.format(filename))

    ##
    #  Writes pitch-marks to the text file.
    #
    #  @param filename name of text file with pitch-marks stored
    #
    def write_file(self, filename):
        # TODO: zapisovat komentar

        try:
            file = open(filename, 'w') if isinstance(filename, str) else filename
            for p in self.__pmlist:
                file.write("%f %f %c\n" % (p.time, p.time, p.type))
            if isinstance(filename, str):
                file.close()
        except IOError:
            raise Exception('File "{}" cannot be written...'.format(filename))


    ##
    # Pickles the current instance to the array of Bytes.
    # @return the array of Bytes containing the pickled data, which can be recovered by #unpickle() method
    #
    def pickle(self) :
        return pickle.dumps(self)

    ##
    # Unpickles the data to the current instance. The method has the same effect as the call of #read_file(),
    # except the data are read from a pickled stream.
    #
    def unpickle(self, data) :
        data = pickle.loads(data)
        # Check the instance
        if not isinstance(data, type(self)) :
            raise TypeError('Unpickling wrong type {}, {} expected', type(data), type(self))
        # "Rewrite" the instance
        self.__dict__ = data.__dict__
        # Get this instance
        return self

    ##
    #  Sets the sequence of pitch-marks into the class. The original pitch-marks holded by the class will be lost!
    #
    #  @param pmlist the array of new pitch-marks to set (array of objects to be passed through pitch-mark factory callable
    #                set through the constructor)
    #  @param self
    #
    def set_pmks(self, pmlist):
        self.__fname    =  None
        self.__last_idx =  0
        self.__pmlist   = [self.__new_pm(*p if not isinstance(p,OnePm) else (p, )) for p in pmlist]

    ##
    # Pitch-mark factory method. This implementation creates new instance of OnePm class. When an extension of OnePm type
    # is required, you are free override this method. However, <b>do not call it directly</b>.
    #
    # @param  pm_obj pitch-mark object to be checked/converted
    # @param  pm_time pitch-mark time (float) passed to the factory method
    # @param  pm_type pitch-mark type (one of OnePm.type_* types) passed to the factory method
    # @return new instance of OnePm or a derived object
    #
    def new_pmk(self, pm_obj = None, pm_time = None, pm_type = None) :
        # Valid objects
        if pm_obj  is not None and isinstance(pm_obj, OnePm) : return pm_obj
        if pm_time is not None and pm_type is not None       : return OnePm(pm_time, pm_type)
        # Unknown
        raise ValueError('Unable to create pitch-mark from obj={}, time={}, type={}'.format(pm_obj, pm_time, pm_type))

    ##
    #  Clears all pitch-marks in the class.
    #
    #  @param self
    #
    def clear(self):
        self.set_pmks([])

    ##
    #  Test whether the index is in the range of the utterance and can be used.
    #
    #  @param idx tested index
    #
    def index_feasible(self, idx):
        return (len(self.__pmlist) > idx) and (0 <= idx)

    ##
    #  Return the time instance of the idx-th pitch-mark
    #
    #  @param idx index of desired pitch-mark
    #
    def get_time(self, idx):
        return self.__pmlist[idx].time
    ##
    #  Return the type of the idx-th pitch-mark
    #
    #  @param idx index of desired pitch-mark
    #
    def get_type(self, idx):
        return self.__pmlist[idx].type

    ##
    #  Set the type of (unit) time boundary shifting (eg. for F<sub>0</sub> calculation)
    #
    def set_shift_type(self):
        self.__shift_type = type

    ##
    #  Return the type of (phonetic) time boundary shifting
    #
    def get_shift_type(self):
        return self.__shift_type

    ##
    #  Test whether the idx-th pitch-mark is voiced
    #
    #  @param idx index of desired pitch-mark
    #
    def is_voiced(self, idx):
        return self.__pmlist[idx].V
    ##
    #  Test whether the idx-th pitch-mark is unvoiced
    #
    #  @param idx index of desired pitch-mark
    #
    def is_unvoiced(self, idx):
        return self.__pmlist[idx].U
    ##
    #  Test whether the idx-th pitch-mark is transitional
    #
    #  @param idx index of desired pitch-mark
    #
    def is_transitional(self, idx):
        return self.__pmlist[idx].T


    ##
    #  Returns new Pm instance, which contains pitch-marks between two given time instances.
    #
    #  @param self
    #  @param time_beg
    #  @param time_end
    #
    def time_slice( self, time_beg=0, time_end=10**100 ):
        # Find and build the class
        bi, ei = self.find_inner_idxs(time_beg, time_end)
        pms   = Pm()
        pms.set_pmks(self[bi:ei+1])
        # Get the boundaries
        return pms

    ##
    #  Calculate the average F<sub>0</sub> in the interval between 2 given indexes or 2 time instances.
    #  <br>
    #  When the unit is unvoiced, the value given by variable <i> invalid_f0 </i> is returned.
    #  <br>
    #  Some units are partly voiced and partly unvoiced; the minimum ratio voiced/unvoiced part for
    #  voiced unit is defined by variable <i> voiced_ratio </i> (default vaue is 0.5).
    #
    #  @param param_type type of following parameters (must be set to constant bound_time or bound_idx)
    #  @param param_begin starting time instance or index (float or int)
    #  @param param_end terminal time instance or index (float or int)
    #
    def get_f0(self, param_type, param_begin, param_end):
        if param_type == Pm.bound_time:

            if param_begin == Pm.idx_continue:  # continue from the last method usage
                idx1 = self.__last_idx
            else:  # seeking the first index from the start of the utterance
                idx1 = self.find_idx(param_begin)
            idx2 = self.find_idx(param_end, idx1)

        elif param_type == Pm.bound_idx:
            idx1 = param_begin
            idx2 = param_end
        else:
            pass

        return self.__count_f0(idx1, idx2)


    ##
    #  Calculate the average F<sub>0</sub> in the interval between 2 given time instances
    #  and return objects of class Segment with resulting F<sub>0</sub> and corrected boundaries.
    #
    #  @param time_begin start time instance (float)
    #  @param time_end terminal time instance (float)
    #
    def get_segment(self, time_begin, time_end):

        if self.__shift_type in {Pm.shift_Left, Pm.shift_Right, Pm.shift_Nearest}:

            if time_begin == Pm.idx_continue:  # continue from the last method usage
                idx1 = self.__last_idx
            else:
                idx1 = self.find_idx_uv(time_begin)
            idx2 = self.find_idx_uv(time_end, idx1)

        else:
            if time_begin == Pm.idx_continue:
                idx1 = self.__last_idx
            else:
                idx1 = self.find_idx(time_begin)
            idx2 = self.find_idx(time_end, idx1)

        f0 = self.__count_f0(idx1, idx2)

        if (self.__shift_type != Pm.shift_none):
            return Segment(self.get_time(idx1), self.get_time(idx2), f0)
        else:
            return Segment(time_begin, time_end, f0)

    ##
    #  Return the sequence of pitch-marks in the interval <code>time_begin, time_end</code> as array [pm.OnePm, pm.OnePm, ...]
    #
    #  @param time_begin start time instance (float)
    #  @param time_end terminal time instance (float)
    #  @param pm_type_incl types of pitch-marks returned in the resulting array (set, optional)
    #  @param pm_type_excl types of pitch-marks not returned in the resulting array (set, optional)
    #
    #  Either <code>pm_type_incl</code> or <code>pm_type_excl</code> should be defined.
    #  If none of them is defined, all pitch-marks will be returned.
    #
    #  Example:
    #  <code>get_pmks( 1.0, 2.0, {OnePm.type_V, OnePm.type_T} )</code>
    #
    def get_pmks( self, time_begin, time_end, pm_type_incl = {OnePm.type_V, OnePm.type_U, OnePm.type_T}, pm_type_excl = {}):
        #### Legacy params handling
        if isinstance(pm_type_incl, (str, list, tuple)) :
           pm_type_incl = set(pm_type_incl)
        if isinstance(pm_type_excl, (str, list, tuple)) :
           pm_type_excl = set(pm_type_excl)
        #### ------

        # Get the expected types
        pm_type_incl = pm_type_incl.difference(pm_type_excl)
        bi,ei        = self.find_inner_idxs(time_begin, time_end)
        # Build the list
        return tuple(pm for pm in self[bi:ei+1] if pm.type in pm_type_incl)

    ##
    #  Return all pitch-marks as array [pm.OnePm, pm.OnePm, ...]
    #
    #  @param pm_type_incl types of pitch-marks returned in the resulting array (set, optional)
    #  @param pm_type_excl types of pitch-marks not returned in the resulting array (set, optional)
    #
    #  Either <code>pm_type_incl</code> or <code>pm_type_excl</code> should be defined.
    #  If none of them is defined, all pitch-marks will be returned.
    #
    #  Example:
    #  <code>get_pmks( pm_type_excl = OnePm.type_U + OnePm.type_T )</code>
    #
    def get_all_pmks(self, pm_type_incl = {OnePm.type_V, OnePm.type_U, OnePm.type_T}, pm_type_excl = {}):
        return self.get_pmks(self[0].time -0.1, self[-1].time + 0.1, pm_type_incl = pm_type_incl, pm_type_excl = pm_type_excl)

    ##
    #  Filter pitch-marks by other pitch-marks.
    #  If a pitch-mark is not present in the another pitch-mark object, it is removed. To express whether the pitch-mark
    #  is present in the another pitch-mark object, isclose() method is applied.
    #
    #  @param other Another pitch-mark object serving as a reference for filtering
    #
    def filter_pmks(self, other):
        self.__pmlist = [p for p in self.__pmlist if p.is_close(other.find(p.time, idx_start=Pm.idx_continue)[1])]

    ##
    #  Return tuple of 2 indices which delimite inner pitch-marks between given time instances
    #
    #  @param time_beg
    #  @param time_end
    #
    def find_inner_idxs(self, time_beg, time_end):
        # Find
        bi,_ = self.find(time_beg, {OnePm.type_V, OnePm.type_U, OnePm.type_T})
        ei,_ = self.find(time_end, {OnePm.type_V, OnePm.type_U, OnePm.type_T}, bi)
        # Get the boundaries
        return (bi,ei)

    ##
    #  Returns tuple of duplicate pitch-marks. The duplicates are pitch-marks with the same type and times close each other.
    #
    #  @param  toler the tolerance in Hz. If set, only pitch-marks closer than 1/tolerance are considered as duplicates, otherwise
    #          the exact time match is required.
    #  @return the list of (refr,duplicate) indexes where the refr is the reference pitch-mark and duplicate is the instance considered
    #          as the diplicate (duplicate is <b>always</b> higher than refer). There may be more items with the same refer value!
    #
    def find_duplicates(self, toler = None):
        dupls = []
        toler = 1.0/toler if toler is not None and toler > 0 else 0.0
        ipm1  = enumerate(self)
        # Search for duplicates
        for r,pm1 in ipm1 :
            for d,pm2 in enumerate(self[r+1:]) :
                # Duplicates must have the same type and time close enough
                if   pm1.type != pm2.type         : break
                elif pm1.time <  pm2.time - toler : break
                # We have duplicate
                d += r+1
                dupls.append((r,d))
                # Skip the reference just processed for further loop. pm1 will remain
                _,_= next(ipm1)
        # Get what find
        return dupls

    ##
    #  Removes the duplicate pitch-marks as find by self.find_duplicates(). It removes the second from the tuples returned.
    #
    #  @param  toler the tolerance in Hz. If set, only pitch-marks closer than 1/tolerance are considered as duplicates, otherwise
    #          the exact time match is required.
    #
    def del_duplicates(self, toler = None):
        for r,d in reversed(self.find_duplicates(toler)) :
            del self[d]

    ##
    #  Returns a list of indeces of isolated pitch-marks. The isolates are pitch-marks with the distance to both the
    #  nearest left and right pitch-mark being greater that the given T0_threshold.
    #
    #  @param  t0_threshold T0 threshold in seconds. Distances higher than this threshold manifest the isolates.
    #  @return the list of oitch-mark indices considered as isolates.
    #
    def find_isolates(self, t0_threshold):
        isos = []
        try:
            if self[1].time - self[0].time > t0_threshold:
                isos.append(1)
        except IndexError:
            raise IndexError('At least two pitch-marks expected! Is there any speech?')
        for idx, p in enumerate(self[1:-1]):
            if (p.time - self[idx-1].time) > t0_threshold and (self[idx+1].time - p.time) > t0_threshold:
                isos.append(idx)
        try:
            if self[-1].time - self[-2].time > t0_threshold:
                isos.append(-1)
        except IndexError:
            raise IndexError('At least two pitch-marks expected! Is there any speech?')
        return isos

    ##
    #  Deletes the isolated pitch-marks as found by self.find_isolates().
    #  @param t0_threshold T0 threshold in seconds. Distances higher than this threshold manifest the isolates.
    def del_isolates(self, t0_threshold):
        for idx in reversed(self.find_isolates(t0_threshold)):
            del self[idx]

    ##
    # Find the nearest pitch-mark of the given type for given time instance (according to pre-set shift type).
    #
    # @param  time time instance for which the pitch mark index is sought (float)
    # @param  type the set of the allowed PMark types (only voiced/unvoiced by default)
    # @param  idx_start starting index for seeking (int, default is 0, if set to <i> idx_continue </i>,
    #         seeking will start from index where last seeking ended)
    # @return (indx,OnePm) tuple with the index and OnePm instance of the pitch-mark found
    #
    def find(self, time, types = {OnePm.type_V, OnePm.type_U}, idx_start = 0):
        # Dummy class holding pitch-marks of the given type with reference to the original pitch-mark time
        class __PM__(object) :
              # Constructor. Sets the pitch-mark and its index in the fill pitch-marks sequence
              def __init__(self, pm, indx) :
                  self.__indx =  indx
                  self.__pm   = pm
              # Gets the original index
              @property
              def indx(self) :
                  return self.__indx
              ## Caller of OnePm.time
              @property
              def time(self) :
                  return self.__pm.time
              ## Caller of OnePm.type
              @property
              def type(self) :
                  return self.__pm.type
              ## Caller of OnePm.__lt__()
              def __lt__(self, obj):
                  return self.__pm.__lt__(obj)
        # ----

        # Build the sequence of the given type
        pmks  = [__PM__(p, i) for i,p in enumerate(self.__pmlist) if idx_start <= i and p.type in types]
        # Find the nearest bigger or equal
        indx  = bisect.bisect_left(pmks, time)
        indx  = min(indx, len(pmks) -1)

        # If equal is expected and not found, error
        if self.__shift_type == Pm.shift_none :
           if   pmks[indx].time != time :
                raise ValueError('Pitch-mark exact to time {} not found, nearest found {} (file {})'.format(time, self.__pmlist[indx].time, self.__fname))
           else :
                return (pmks[indx].indx,self[pmks[indx].indx])

        # Nearest lower
        indxl = max(indx -1, 0)
        # Check nearest
        if   self.__shift_type == Pm.shift_nearest :
             return (pmks[indx].indx, self[pmks[indx].indx]) if (pmks[indx].time - time) < (time - pmks[indxl].time) else (pmks[indxl].indx, self[pmks[indxl].indx])
        elif self.__shift_type == Pm.shift_right :
             return (pmks[indx].indx, self[pmks[indx].indx])
        elif self.__shift_type == Pm.shift_left :
             return (pmks[indxl].indx,self[pmks[indxl].indx])
        else :
             raise ValueError('No nearest pitch-mark found')

    ##
    #  Find the nearest pitch-mark for given time instance (according to pre-set shift type).
    #
    #  @param time time instance for which the pitch mark index is sought (float)
    #  @param idx_start starting index for seeking (int, default is 0, if set to <i> idx_continue </i>,
    #    seeking will start from index where last seeking ended)
    #  @param skip_T if <code>True</code>, transitional pitch-marks will be skipped during the search
    #
    def find_idx(self, time, idx_start = 0, skip_T = True):
        idx,_  = self.find(time, {OnePm.type_V, OnePm.type_U} if skip_T else {OnePm.type_V, OnePm.type_U, OnePm.type_T}, idx_start)
        return   idx

    ##
    #  Find the nearest pitch-mark for given time instance (according to pre-set shift type).
    #  On boundaries between voiced and unvoiced unit try to find first/last unvoiced pitch-mark
    #
    #  @param time time instance for which the pitch mark index is sought (float)
    #  @param idx_start starting index for seeking (int, default is 0, if set to <i> idx_continue </i>,
    #    seeking will start from index where last seeking ended)
    #
    def find_idx_uv(self, time, idx_start = 0):

        idx = self.find_idx(time, idx_start)
        if self.is_voiced(idx):
            idx2 = idx - 1
            while self.index_feasible(idx2) and (self.get_time(idx2) > time - Pm.uv_boundary_tol):
                if self.is_unvoiced(idx2):
                    idx = idx2
                    break
                idx2 -= 1

            else:
                idx2 = idx + 1
                while self.index_feasible(idx2) and (self.get_time(idx2) < time + Pm.uv_boundary_tol):
                    if self.is_unvoiced(idx2):
                        idx = idx2
                        break
                    idx2 += 1

        elif self.is_unvoiced(idx):
            idx2 = idx - 1
            while self.index_feasible(idx2) and (self.get_time(idx2) > time - Pm.uv_boundary_tol):
                if self.is_voiced(idx2):
                    idx = idx2+1
                    break
                idx2 -= 1

            else:
                idx2 = idx + 1
                while self.index_feasible(idx2) and (self.get_time(idx2) < time + Pm.uv_boundary_tol):
                    if self.is_voiced(idx2):
                        idx = idx2-1
                        break
                    idx2 += 1

        self.__last_idx = idx
        return idx

    ##
    #  Find the nearest pitch-mark for given time instance (according to pre-set shift type) and returns its instance.
    #
    #  @param time time instance for which the pitch mark index is sought (float)
    #  @param idx_start starting index for seeking (int, default is 0, if set to <i> idx_continue </i>,
    #    seeking will start from index where last seeking ended)
    #  @param skip_T if <code>True</code>, transitional pitch-marks will be skipped during the search
    #  @return the instance of pitch-mark found (pm.OnePm)
    #  @see pm.Pm.find_idx()
    #  @deprecated
    #
    def find_pmk(self, time, idx_start = 0, skip_T = True):
        return self.__pmlist[self.find_idx(time, idx_start, skip_T)]

    ##
    #  Find the nearest pitch-mark for given time instance (according to pre-set shift type).
    #
    #  @param time time instance for which the pitch mark index is sought (float)
    #  @param idx_start starting index for seeking (int, default is 0, if set to <i> idx_continue </i>,
    #    seeking will start from index where last seeking ended)
    #  @return the instance of pitch-mark found (pm.OnePm)
    #  @see pm.Pm.find_idx_uv()
    #  @deprecated
    #
    def find_pmk_uv(self, time, idx_start = 0):
        return self.__pmlist[self.find_idx_uv(time, idx_start)]

    ##
    #  Shifts the given pitch-mark the given number of steps
    #
    #  @param idx_start the index to start from
    #  @param steps the number of PMarks to shift (<i>-n</i> shifts <i>n</i> steps to the left in the
    #    sequence, <i>n</i> shifts <i>n</i> steps to the right.
    #  @param skip_T if <code>True</code>, all transitional pitch-marks will be skipped during the shift
    #
    def shift_idx(self, idx_start, steps, skip_T = True) :
        end = idx_start + steps
        step = int(steps/abs(steps)) # 1 or -1
        while idx_start != end :
            idx_start += step
            # Out of range
            if idx_start < 0 :
                raise IndexError('Pitch-mars sequence underflow')
            if idx_start >= len(self) :
                raise IndexError('Pitch-mars sequence overflow')

            # Skip T
            if skip_T and self.__pmlist[idx_start].T :
                end += step
        return end


    ##
    # Return the name of file from which the pitch-marks were read.
    #
    @property
    def source(self):
        return self.__fname

    ##
    # Return the name of loaded text file with pitch-marks.
    # @deprecated
    #
    def get_file_name(self):
        print('Pm.get_file_name() is deprecated, use Pm.source property', file = sys.stderr)
        return self.source

    ##
    # Appends a new pitch-mark or a sequence of pitch-marks.
    #
    # @param self
    # @param items a pitch-mark array of new pitch-marks to append (Pm instance or array of objects to be
    #        passed through Pm.new_pmk() pitch-mark factory)
    #
    def append(self, items):
        if isinstance( items, (list, tuple, Pm)):
           # Add one-by-one
           for p in items : self.append(p)
        else :
           # Build the item
           p = self.__new_pm(items)
           p = self.__chk_pm(p, self.__pmlist[-1].time if len(self) > 0 else 0.0, sys.float_info.max)
           # Set it
           self.__pmlist.append(p)

    ##
    # Insert new pitch-mark or a sequence of pitch-marks to the given position.
    #
    # @param items an pitch-mark or array of new pitch-marks to insert (Pm instance or array of objects to be
    #        passed through Pm.new_pmk() pitch-mark factory)
    # @param idx    Index of a pitch-mark to insert before (like in standard insert function)
    #
    def insert(self, idx, items) :
        if isinstance(items, (list, tuple, Pm)):
           # Add one-by-one. It is, for up to tens on pitch-marks inserted, not slower than building a auxiliary array
           # of pitch-marks which is then added in one step.
           for i,p in enumerate(items) : self.insert(idx+i, p)
        else :
           # Build the item
           p = self.__new_pm(items)
           p = self.__chk_pm(p, self.__pmlist[idx-1].time if idx > 0         else 0.0,
                                self.__pmlist[idx].time   if idx < len(self) else sys.float_info.max)
           # Set it
           self.__pmlist.insert(idx, p) if idx < len(self) else self.__pmlist.append(p)

    ##
    # Sets new pitch-mark or a sequence of pitch-marks to the given position, given by their times. There is
    # one requirement on the sequence of Pitch-marks placed - the whole sequence must be placeable between two
    # existing pitch-marks!
    #
    # @param items an pitch-mark or array of new pitch-marks to insert (Pm instance or an array of Pm objects!)
    # @param idx    Index of a pitch-mark to insert before (like in standard insert function)
    #
    def update(self, items) :
        # Find the index if the nearest highed pitch-mark
        if   isinstance(items, (list, tuple, Pm)): idx = bisect.bisect_left(self.__pmlist, items[0].time)
        elif isinstance(items, OnePm)            : idx = bisect.bisect_left(self.__pmlist, items.time)
        else                                     : raise ValueError('Don\'t know how to get time from {} object'.format(items))
        # Call the insert method
        self.insert(idx, items)

    ##
    # The method enabling index-like access into the array of pitch-marks.
    #
    # @param  index the index of the required pitch-mark (int)
    # @param  self
    # @return OnePm object
    #
    def __getitem__(self, index):
        return self.__pmlist[index]
    ##
    # The method enabling index-like setting into the array of pitch-marks. The original item is replaced!
    #
    # @param  index the index of the required pitch-mark (int)
    # @param  pm_obj an pitch-mark to set (Pm instance or an object to be passed through Pm.new_pmk()
    #         pitch-mark factory)
    # @param  self
    #
    def __setitem__(self, index, pm_obj):
        # Convert and check the pitch-mark
        pm_obj = self.__new_pm(pm_obj)
        pm_obj = self.__chk_pm(pm_obj, self[index -1].time if index> 0 else 0.0,
                                       self[index   ].time)
        # Set it
        self.__pmlist[index] = pm_obj

    ##
    #  The metod for deleting pitch-mark given by index
    #
    # @param self
    # @param index the index of the required pitch-mark (int)
    #
    def __delitem__(self, index):
        if (index < len( self.__pmlist)):
            del self.__pmlist[index]

    ##
    # The method getting the number of pitch-marks in the class
    #
    # @param  self
    #
    def __len__(self):
        return len(self.__pmlist)

    ##
    # The method printing user-readable information about the pitch-mark object
    #
    # @param  self
    #
    def __repr__(self):
        return "[Array of %8d pitch-marks, read from: %s]" % (len(self.__pmlist), self.__fname)

    ##
    #  Process the content of pitch-mark text file.
    #  @param source iterator through the lined to read pitch-marks from
    #
    def __process_file(self, source):
        last = -0.0001
        # Read from the source
        for line in source:
            # Build the pitch-mark object
            line = line.split()
            time = float(line[1])
            pm   = self.__new_pm(time, line[2])
            pm   = self.__chk_pm(pm, last, sys.float_info.max)
            # Store and set the last PM
            self.__pmlist.append(pm)
            last = time

    ##
    #  Counts the average F<sub>0</sub> in the interval between 2 indexes.
    #
    def __count_f0(self, idx1, idx2):

        vpm_count = 0
        f0_sum = 0
        v_signal = 0
        u_signal = 0

        time1 = self.get_time(idx1)

        idx = idx1
        while idx2 > idx:

            if self.is_transitional(idx):
                idx = idx + 1
                continue

            time2 = self.get_time(idx + 1)
            pm_dist = time2 - time1

            if self.is_voiced(idx) and self.is_voiced(idx+1):

                if pm_dist <= Pm.voiced_pm_dist:
                    f0_sum = f0_sum + 1/pm_dist
                    v_signal = v_signal + pm_dist
                else:
                    u_signal = u_signal + pm_dist
                vpm_count = vpm_count + 1
            else:
                u_signal = u_signal + pm_dist

            time1 = time2
            idx = idx + 1

        if vpm_count > 0 and v_signal/float(v_signal + u_signal) >= Pm.voiced_ratio:
            f0 = f0_sum/vpm_count  # frequency of voiced unit
        else:
            f0 = Pm.invalid_f0 # unvoiced unit has no f0 frequency

        return f0

    ##
    # Checks if pitch-mark time is between the giben time interval. Throws ValueError exception,
    # when it is not. The method is primarily used to check the pitch-marks ordering.
    #
    # @param  pm_obj pitch-mark object to be checked
    # @param  time_min the minimum time value allowed for the pitch-mark
    # @param  time_max the maximum time value allowed for the pitch-mark
    # @return the pm_obj, when correct
    #
    def __chk_pm(self, pm_obj, time_min, time_max) :
        if pm_obj.time >= time_min and pm_obj.time <= time_max : return pm_obj
        raise ValueError('Invalid pitch-mark instance [{}], out of time intervals <{},{}>'.format(pm_obj, time_min, time_max))
    ##
    # Creates new PMark object, using Pm.new_pmk() method. The arguments may either be a single object passed
    # to Pm.new_pmk() factory, or a tuple of (time, type) values passed to Pm.new_pmk() as the individual time/type
    # attributes
    #
    def __new_pm(self, *args) :
        # Create the instance
        if   len(args) == 1 : pm_obj = self.new_pmk(pm_obj = args[0], pm_time = None,    pm_type = None)
        elif len(args) == 2 : pm_obj = self.new_pmk(pm_obj = None,    pm_time = args[0], pm_type = args[1])
        else :
             raise ValueError('Don\'t know how to get pitch-mark time/type from {}'.format(args))
        # Check the instance
        if   not isinstance(pm_obj, OnePm) :
             raise TypeError('Invalid pitch-mark instance [{}] created, [{}] expected'.format(type(pm_obj), OnePm))
        # Get it
        return pm_obj



##
# Procedure computing the mean F0 from the list (or tuple) of pm.OnePm classes.
# Only pitch-marks of given type are taken into account.
#
# @param  pmarks the sequence of pitch-marks (list or tuple of pm.OnePm instances)
# @param  typ    the type of pitch-marks computed (pm.OnePm.type_V or pm.OnePm.type_U)
#                The default is voiced type.
# @return the mean F0 value (float), or -1 when it cannot be computed
#
def counf_f0(pmarks, typ = OnePm.type_V):

    sum_F0 = 0
    num_F0 = 0
    # Process all pitch-marks
    for i in range(1, len(pmarks)) :
        if (pmarks[i -1].type == typ and pmarks[i].type == typ) :
            sum_F0 += 1.0 / (pmarks[i].time - pmarks[i -1].time)
            num_F0 += 1

    # Return the value
    if num_F0 == 0 :
       return -1.0
    else :
       return sum_F0 / num_F0

##
# Procedure computing the mean T0 from the list (or tuple) of pm.OnePm classes.
# Only pitch-marks of given type are taken into account.
#
# @param  pmarks the sequence of pitch-marks (list or tuple of pm.OnePm instances)
# @param  typ    the type of pitch-marks computed (pm.OnePm.type_V or pm.OnePm.type_U)
#                The default is voiced type.
# @return the mean T0 value (float), or -1 when it cannot be computed
#
def counf_t0(pmarks, typ = OnePm.type_V):

    sum_T0 = 0
    num_T0 = 0
    # Process all pitch-marks
    for i in range(1, len(pmarks)) :
        if (pmarks[i -1].type == typ and pmarks[i].type == typ) :
            sum_T0 += pmarks[i].time - pmarks[i -1].time
            num_T0 += 1

    # Return the value
    if num_T0 == 0 :
       return -1.0
    else :
       return sum_T0 / num_T0

