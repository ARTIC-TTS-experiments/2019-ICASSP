#
# Module implementing pitch-mark comparison function as the replacement of StatisticsPM binary implemented in C++
# The code is based on the StatisticsPM implementation from 8.4.2002 (rev. 1447)

from  pm import OnePm,Pm

##
# Pitch-mark object. It extends pm.OnePm with the ability to store used-defined attributes
# to allow effective comparison.
#
class PM(OnePm, dict) :

      ## The local F0 for this PMark, computed from its neighbours
      key_localT0    = 'localT0'
      ## The nearest PMark to the current from the other sequence
      key_nearest    = 'nearest'
      ## The type of sequence the pitch-mark belongs to (may be one of (PM.key_seqRefr, pm.key_seqTest) values
      key_seqType    = 'sequence'
      ## The reference sequence type
      key_seqRefr    = 'refr'
      ## The tested sequence type
      key_seqTest    = 'test'

      ##
      # Factory method getting the instance of PM pitch-mark representation.
      #
      @staticmethod
      def factory(pm_obj, pm_time, pm_type) :
          # Object given
          if pm_obj is not None :
             if   isinstance(pm_obj, PM)    : return pm_obj
             elif isinstance(pm_obj, OnePm) : return PM(pm_obj.time, pm_obj.type)
          if pm_time is not None and pm_type  is not None :
             return PM(pm_time, pm_type)
          # Unknown
          raise ValueError('Unable to create pitch-mark from obj={}, time={}, type={}'.format(pm_obj, pm_time, pm_type))
      ##
      # Extension PM.factory() factory method, getting the instance of the PM pitch-mark representation
      # of tested pitch-mark.
      #
      @staticmethod
      def factory_test(pm_obj, pm_time, pm_type) :
          pm                 = PM.factory(pm_obj, pm_time, pm_type)
          pm[pm.key_seqType] = pm.key_seqTest
          return pm
      ##
      # Extension PM.factory() factory method, getting the instance of the PM pitch-mark representation
      # of reference pitch-mark.
      #
      @staticmethod
      def factory_refr(pm_obj, pm_time, pm_type) :
          pm                 = PM.factory(pm_obj, pm_time, pm_type)
          pm[pm.key_seqType] = pm.key_seqRefr
          return pm


##
# Reference pitch-mark holder. It defines new constructor, setting the sequence of pitch-marks.
# DO NOT USE THIS INSTANCE DIRECTLY!
#
class PMseq(Pm) :
      # Constructor
      # @param pmks the sequence of pitch-marks to set
      def __init__(self, pmks) :
          super(PMseq, self).__init__(shift_type = Pm.shift_nearest)
          super(PMseq, self).set_pmks(pmks)
      # Override of pm.Pm.new_pmk()
      def new_pmk(self, pm_obj = None, pm_time = None, pm_type = None) :
          raise RuntimeError('Do not create {} instance!'.format(type(self)))
##
# Reference pitch-mark holder. It extends PMseq, overriding the pm.Pm.new_pmk() method, which in turn calls
# PM.factory_refr().
#
class PM_Refr(PMseq) :
      # Override of pm.Pm.new_pmk()
      def new_pmk(self, pm_obj = None, pm_time = None, pm_type = None) :
          return PM.factory_refr(pm_obj = pm_obj, pm_time = pm_time, pm_type = pm_type)
##
# Tested pitch-mark holder. It extends PMseq, overriding the pm.Pm.new_pmk() method, which in turn calls
# PM.factory_test().
#
class PM_Test(PMseq) :
      # Override of pm.Pm.new_pmk()
      def new_pmk(self, pm_obj = None, pm_time = None, pm_type = None) :
          return PM.factory_test(pm_obj = pm_obj, pm_time = pm_time, pm_type = pm_type)


##
# Pitch-mark comparison holder
#
class PM_Compare(object) :

      ##
      # The name of argument accepted by PM_Compare constructor. The argument sets value get by #diff_abs()
      # @see #diff_abs()
      #
      conf_diff_abs    = 'diff_abs'
      ##
      # The name of argument accepted by PM_Compare constructor. The argument sets value get by #diff_t0()
      # @see #diff_t0()
      #
      conf_diff_t0     = 'diff_t0'
      ##
      # The name of argument accepted by PM_Compare constructor. The argument specifies, if the unsure pitch-marks
      # should be treated as regular voiced pitch-marks or if they whould be ignored
      #
      conf_incl_unsure = 'incl_unsure'

      ##
      # The name of key (in output data) containing the pm.OnePm instance of a reference pitch-mark.
      #
      outp_refr_pm     = 'pm_refr'
      ##
      # The name of key (in output data) containing the pm.OnePm instance of a tested pitch-mark.
      #
      outp_test_pm     = 'pm_test'
      ###
      # The name of key (in output data) containing the distance from reference to tested pitch-mark instances.
      #
      outp_dist_pm     = 'pm_dist'
      ###
      # The name of key (in output data) containing the local pitch-perid assigned to the reference pitch-mark
      #
      outp_localT0     = PM.key_localT0


      ##
      # Constructor
      # @param diff_abs the value get by PM_Compare.diff_abs property
      # @param diff_t0 the value get by PM_Compare.diff_f0 property
      # @param incl_unsure if <code>True</code> the pitch-marks marked as <i>unsure</i> (type = "?") will be treated
      #        as all the other voiced pitch-marks. Otherwise, they are ignored.
      #
      def __init__(self, diff_abs = None, diff_t0 = None, incl_unsure = False, **kwargs) :
          # Comparison config
          self.__PMdiff =  None
          self.__config = {self.conf_diff_abs :  float(diff_abs)         if diff_abs is not None else diff_abs,
                           self.conf_diff_t0  : (float(diff_t0) / 100.0) if diff_t0  is not None else diff_t0,
                          }
          # Get the comparison according to the config
          if   self.diff_abs is None and isinstance(self.diff_t0,  (int, float)) and self.diff_t0  >= 0.0 :
               self.__PMdiff =  self.compute_diff_T0
          elif self.diff_t0  is None and isinstance(self.diff_abs, (int, float)) and self.diff_abs >= 0.0 :
               self.__PMdiff =  self.compute_diff_abs
          else :
               raise ValueError('No difference method configured correctly: {}'.format(self.__config))

          # Variables
          self.__pm_refr     =  Pm()
          self.__pm_test     =  Pm()
          self.__dels        =  None
          self.__inss        =  None
          self.__shft        =  None
          self.__incl_unsure =  incl_unsure


      ##
      # If the distance of reference to tested pitch-mark is supposed to be measured as a absolute difference of their
      # times, this method returns the maximum difference under which the test and reference pitch-marks are supposed
      # to be equal.
      #
      # @return the maximum difference of pitch-mark times, or <code>None</code> if this measure should not be used
      # @see    #compute_diff_abs()
      #
      @property
      def diff_abs(self) : # -> float:
          return self.__config.get(self.conf_diff_abs, None)
      ##
      # If the distance of reference to tested pitch-mark is supposed to be measured as a percentage of pitch-period
      # local to the reference pitch-mark, this method returns the maximum percentage (in range <0.0,1.0>) of the
      # pitch-period under which the test and reference pitch-marks are supposed to be equal.
      #
      # @return the percentage of the reference local pith-period used to compare the distance, or <code>None</code>
      #         if this measure should not be used
      # @see    #compute_diff_t0()
      #
      @property
      def diff_t0(self) : # -> float:
          return self.__config.get(self.conf_diff_t0,  None)

      ##
      # Compares two sequences of pitch-marks read from files. It reads the files and calls PM_Compare.compare_pmSeq()
      # method.
      #
      # @param  pm1 the name of file with sequence of reference pitch-marks
      # @param  pm2 the name of file with sequence of tested pitch-marks
      # see     PM_Compare.compare_pmSeq() for more details.
      #
      def compare_files(self, pm_refr, pm_test) :
          pm_refr = Pm(filename = pm_refr)
          pm_test = Pm(filename = pm_test)
          # Compare the sequences
          return self.compare_pmSeq(pm_refr, pm_test)

      ##
      # Compares two sequences of pitch-marks. The method compares the pitch-mark sequences (tested to reference), but
      # does not return any results. To get them, call PM_Compare.inserted(), PM_Compare.deleted(), PM_Compare.shifted(),
      # PM_Compare.tested(), PM_Compare.reference() or PM_Compare.score_Lev() methods.
      #
      # The inputs are pitch-mark holder instances, e.g. Pm class, or a sequence of OnePm instances.
      #
      # @param pm1 a sequence of reference pitch-marks
      # @param pm2 the sequence of tested pitch-marks
      #
      def compare_pmSeq(self, pm_refr, pm_test) :
          # Check the instance
          if not isinstance(pm_refr, (Pm, list, tuple)) :
             raise TypeError('Invalid pitch-marks holder {}!'.format(pm_refr))
          if not isinstance(pm_test, (Pm, list, tuple)) :
             raise TypeError('Invalid pitch-marks holder {}!'.format(pm_test))

          # Clear deletes, inserts and shifts
          self.__dels    = None
          self.__inss    = None
          self.__shft    = None
          # Read the PMarks
          self.__pm_refr = PM_Refr(pm_refr)
          self.__pm_test = PM_Test(pm_test)

          # Include unsure in reference (replace the unsure by V type)
          for i,p in reversed(list(enumerate(self.__pm_refr))) :
              if p.type == '?' :
                 if self.__incl_unsure :     self.__pm_refr[i] = PM(p.time, p.type_V)
                 else                  : del self.__pm_refr[i]


          # Build the nearest pairs
          # - for each PMark in pm_refr find the nearest in pm_test
          # - for each PMark in pm_test find the nearest in pm_refr
          self.__build_pairs(self.__pm_refr, self.__pm_test)
          self.__build_pairs(self.__pm_test, self.__pm_refr)
          # Compute local F0 for the reference sequence
          self.__local_T0(self.__pm_refr)

      ##
      # Returns the sequence of inserted pitch-marks. The inserted pitch-mark is that which occurs in
      # the reference sequence but not in the tested sequence (thus is must be inserted into the
      # tested to match the reference)
      #
      # @param  items the set of items to be filled into the output sequence (one of PM_Compare.outp_* keys)
      # @return new sequence of inserted pitch-marks (each call builds new sequence); items are dictionaries
      #         with the required items filled
      #
      def inserted(self, items = {outp_refr_pm}) :
          # Build the sequence if not built yet
          if self.__inss is None :
             self.__build_SDI()
          # Build the output items
          return [self.__build_pmdata(pm, items) for pm in self.__inss]
      ##
      # Returns the sequence of deleted pitch-marks. The deleted pitch-mark is that which occurs in
      # the tested sequence but not in the reference sequence (thus is must be deleted from the
      # tested to match the reference)
      #
      # @param  items the set of items to be filled into the output sequence (one of PM_Compare.outp_* keys)
      # @return new sequence of deleted pitch-marks (each call builds new sequence); items are dictionaries
      #         with the required items filled
      #
      def deleted(self, items = {outp_test_pm}) :
          # Build the sequence if not built yet
          if self.__dels is None :
             self.__build_SDI()
          # Build the output items
          return [self.__build_pmdata(pm, items) for pm in self.__dels]
      ##
      # Returns the sequence of shifted pitch-marks. The shifted pitch-mark is that which occurs in
      # the tested sequence but its distance to the nearest reference is out of the given limit defined
      # by the distance measure value set through the constructor
      #
      # @param  items the set of items to be filled into the output sequence (one of PM_Compare.outp_* keys)
      # @return new sequence of shifted pitch-marks (each call builds new sequence); items are dictionaries
      #         with the required items filled
      #
      def shifted(self, items = {outp_refr_pm, outp_test_pm}) :
          # Build the sequence if not built yet
          if self.__shft is None :
             self.__build_SDI()
          # Build the output items
          return [self.__build_pmdata(pm[0], items) for pm in self.__shft]

      ##
      # Returns the sequence of reference pitch-marks (optionally with additional attributes)
      #
      # @param  items the set of items to be filled into the output sequence (one of PM_Compare.outp_* keys)
      # @return new sequence of paired pitch-marks (each call builds new sequence); items are dictionaries
      #         with the required items filled
      #
      def reference(self, items = {outp_refr_pm, outp_test_pm}) :
          return  [self.__build_pmdata(pm, items) for pm in self.__pm_refr]
      ##
      # Returns the sequence of tested pitch-marks (optionally with additional attributes)
      #
      # @param  items the set of items to be filled into the output sequence (one of PM_Compare.outp_* keys)
      # @return new sequence of paired pitch-marks (each call builds new sequence); items are dictionaries
      #         with the required items filled
      #
      def tested(self, items = {outp_refr_pm, outp_test_pm}) :
          return [self.__build_pmdata(pm, items) for pm in self.__pm_test]

      ##
      # Returns the value of Shift-Insert-Delete score computed through <i>Levenshtein distance</i>.
      #
      def score_Lev(self) :
          # Get the base stats
          shft = len(self.__shft) if self.__shft else 0
          inss = len(self.__inss) if self.__inss else 0
          dels = len(self.__dels) if self.__dels else 0
          # The number of pitch-marks is equal to the maximum number of voiced pitch-marks
          pms  = max(len([None for p in self.__pm_refr if p.V]),
                     len([None for p in self.__pm_test if p.V]))
          # Not computed?
          if shft + inss + dels + pms <= 0 :
             raise ValueError('No pitch-marks evaluated!')
          # Score
          return (pms - dels - inss - shft) * 100.0 / pms;


      ##
      # Compares the <code>dist</code> distance against the distance get by #diff_abs().
      #
      # @param pm the instance of PM holding the tested pitch-mark
      # @param dist the distance of the tested pitch-mark to the nearest reference pitch-mark
      # @see   #diff_abs()
      #
      def compute_diff_abs(self, pm, dist) :
          # The pitch-mark must be tested
          if pm.key_seqType not in pm or pm[pm.key_seqType] != pm.key_seqTest :
             raise KeyError('Not tested pitch-mark: {}'.format(pm))
          # Compare
          return dist > self.diff_abs

      ##
      # Compares the <code>dist</code> distance against the distance obtained as the percentage of the
      # pitch-period local to the corresponding reference pitch-mark.
      #
      # @param pm the instance of PM holding the tested pitch-mark
      # @param dist the distance of the tested pitch-mark to the nearest reference pitch-mark
      # @see   #diff_t0()
      #
      def compute_diff_T0(self, pm, dist) :
          # Get the nearest PM (must be reference), the nearest holds (PM, index) tuple
          rf,_ = pm[pm.key_nearest]

          # The pitch-mark must be tested
          if pm.key_seqType not in pm or pm[pm.key_seqType] != pm.key_seqTest :
             raise KeyError('Not tested pitch-mark: {}'.format(pm))
          # rf must have local F0 computed
          if rf.key_localT0 not in rf :
             raise KeyError('There is no {} computed for {}'.format(rf.key_localT0, rf))

          # Compare
          return dist > (rf[rf.key_localT0] * self.diff_t0)

      ##
      # Builds the internal variables holding the sequence of <i>shifted</i>, <i>deleted</i> and <i>inserted</i>
      # pitch-marks.
      #
      # @see Compare.shifted() for the description of what <i>shifted</i> pitch-mark represents.
      # @see Compare.deleted() for the description of what <i>deleted</i> pitch-mark represents.
      # @see Compare.inserted() for the description of what <i>inserted</i> pitch-mark represents.
      #
      def __build_SDI(self) :
          self.__dels = []
          self.__inss = []
          self.__shft = []
          # Process the sequences
          for pm_r,pm_t in PM_Compare.__build_sameref(self.__pm_refr, self.__pm_test) :
              # No tested assigned to the reference - operation insert
              if   not pm_t :
                   self.__inss.append(pm_r)
              # Single assigned - it is shift
              elif len(pm_t) == 1 :
                   self.__shft.extend((p,d) for p,d in pm_t if self.__PMdiff(p,d))
              # Multiple assigned - some of them are deleted, the nearest is shifted
              else :
                   # From the candidates to delete, keep that nearest to the reference. That will have
                   # shift operation
                   d_min = min(d for p,d in pm_t)
                   s_min = d_min + 0.0001
                   # Put all those except the nearest
                   self.__dels.extend( p    for p,d in pm_t if d > d_min)
                   self.__shft.extend((p,d) for p,d in pm_t if d < s_min and self.__PMdiff(p,d))

      ##
      # Computes the local pitch-periods assigned to each pitch-mark. For <i>i</i>-th pitch-mark, the
      # period is computed as the <code>mean(T0(i-1:i+1)</code> when pitch-marks <i>i-1,i,i+1</code> are
      # voiced. If <i>i</i>-1 or <i>i</i>+1 pitch-marks are not voiced, the T0(i-1:i) or (i:i+1) period is
      # taken.
      #
      @staticmethod
      def __local_T0(pms) :
          # Run through (previous, current, next) triples
          for p,c,n in zip(pms[0:-2], pms[1:-1], pms[2:]) :
              # Skip, if the current is not voiced
              if not c.V :
                   c[c.key_localT0] = -1
              # If both previous and next are voiced, compute the local F0
              elif p.V and n.V :
                   c[c.key_localT0] = (n.time - p.time) / 2.0
              elif p.V and c.V :
                   c[c.key_localT0] = (c.time - p.time)
              elif c.V and n.V :
                   c[c.key_localT0] = (n.time - c.time)
              else :
                   c[c.key_localT0] = -1.0
                   # Single V surrounded by C?

          # Get the sequence back
          return pms

      ##
      # Couples the nearest pitch-marks. For each pitch-mark in the first sequence finds the nearest pitch-mark
      # in the second sentence (of the same type). The nearest is stored under PM.key_nearest key.
      #
      # @param pms1 the pitch-mark sequence for which the nearest are filled
      # @param pms2 the pitch-mark sequence in which the nearest are searched
      #
      @staticmethod
      def __build_pairs(pms1, pms2) :
          # Run pmarks in PMS1
          for p1 in pms1 :
              # Reset the nearest key
              p1[p1.key_nearest] = (None,-1)
              # Ignore U/T pitch-marks
              if not p1.V :
                 continue
              # Find the nearest in pms2 to the current pmark (from pms1)
              i,p2 = pms2.find(p1.time, {p1.type, })
              # Set the nearest and its index
              p1[p1.key_nearest] = (p2, i)

      ##
      # Returns the dictionary containing info about a pitch-mark. The dictionary contains keys
      # passed through <i>items</i> (one of PM_Compare.outp_* keys).
      #
      # @param  pm the pith-mark to build the information for
      # @oaram  items a collection of keys to fill within the output dictionary
      # @return dictionary with values for the required keys
      #
      @staticmethod
      def __build_pmdata(pm, items = (outp_refr_pm, outp_test_pm)) :
          # Prepare the data
          s = pm.get(pm.key_seqType, '')
          v = {}
          # Decide about reference and tested pitch-mark
          if   s == pm.key_seqRefr : (rp,tp) = (pm, pm.get(pm.key_nearest, (None, None))[0])
          elif s == pm.key_seqTest : (tp,rp) = (pm, pm.get(pm.key_nearest, (None, None))[0])
          else                     :  raise ValueError('Unable to decide about tested and reference pitch-mark for {}'.format(pm))

          # Set individual items
          for  i in items :
               if i == PM_Compare.outp_test_pm : v[i] = None if tp   is  None   else  OnePm(tp.time, tp.type)
               if i == PM_Compare.outp_refr_pm : v[i] = None if rp   is  None   else  OnePm(rp.time, rp.type)
               if i == PM_Compare.outp_localT0 : v[i] = -1   if rp   is  None   else rp.get(rp.key_localT0, -1)
               if i == PM_Compare.outp_dist_pm : v[i] = -1   if None in {tp,rp} else    abs(rp.time - tp.time)
          # Get the dictionary
          return v

      ##
      # Builds the sequences of tested pitch-marks which all refer to the same reference pitch-mark.
      # For illustration, having the sequences:
      #
      #                          1     2     3   4    5   6   7
      #            pm_refr: .... |     |     |   |    |   |   |  .....
      #            pm_test: .... | | | |    |  | | |  |      |   .....
      #                          1 2 3 4    5  6 7 8  9      10
      #    points to refer no.:  1 1 2 2    3  4 4 4  5      7
      #
      # The method will build sequences:
      #                          (1,[1,2]), (2,[3,4]), (3,[5]), (4,[6,7,8]), (5,[9]), (6,[]), (7,[10])
      #
      # The method is implemented as a generator!
      #
      # @param  pm_refr the sequence of the reference pitch-marks
      # @param  pm_test the sequence of the tested pitch-marks
      # @return each generator step gets <i>PM_ref</i> and a collection of (PM_test,dist) tuples, where
      #         PM_ref is the instance of the reference pitch-mark to which several PM_test instances are
      #         "assigned" as the nearest, having <i>dist</i> distance to the <i>PM_ref</i>.
      #
      @staticmethod
      def __build_sameref(pm_refr, pm_test) :
          # Output array
          pm_out  = []
          y       = 0
          # Go through the reference pitch-marks and search for all the tested pointing to this reference
          # pitch-mark
          for r,pm_r in enumerate(pm_refr) :
              # Ignore, if the nearest is not defined
              if not pm_r.get(pm_r.key_nearest, (None,   -1))[0] :
                 continue

              # The nearest tested and the nearest reference to the tested
              pm_t,t =  pm_r[pm_r.key_nearest]
              pm_x,x =  pm_t[pm_t.key_nearest]

              # The sequence of tested pointing to pm_r reference
              pm_out.clear()
              # Now go through the tested and search for all pointing to the reference
              for pm_t   in pm_test[y:] :
                  pm_x,x =  pm_t.get(pm_t.key_nearest, (None, -1))
                  # If tested points to the another (higher) reference, leave, if points to the same reference, add it to the array
                  if   x  > r :  break
                  elif x == r :  pm_out.append((pm_t, abs(pm_t.time - pm_r.time)))
              # Get what collected
              yield (pm_r, pm_out)
              y = t
