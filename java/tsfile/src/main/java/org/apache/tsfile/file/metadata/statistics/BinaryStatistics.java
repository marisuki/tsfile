/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.tsfile.file.metadata.statistics;

import org.apache.tsfile.common.conf.TSFileConfig;
import org.apache.tsfile.enums.TSDataType;
import org.apache.tsfile.exception.filter.StatisticsClassException;
import org.apache.tsfile.utils.Binary;
import org.apache.tsfile.utils.RamUsageEstimator;
import org.apache.tsfile.utils.ReadWriteIOUtils;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.Objects;

import static org.apache.tsfile.utils.RamUsageEstimator.sizeOfCharArray;

/** Statistics for string type. */
public class BinaryStatistics extends Statistics<Binary> {

  public static final long INSTANCE_SIZE =
      RamUsageEstimator.shallowSizeOfInstance(BinaryStatistics.class)
          + 2 * RamUsageEstimator.shallowSizeOfInstance(Binary.class);

  private static final Binary EMPTY_VALUE = new Binary("", TSFileConfig.STRING_CHARSET);

  private Binary firstValue = EMPTY_VALUE;
  private Binary lastValue = EMPTY_VALUE;

  @Override
  public TSDataType getType() {
    return TSDataType.TEXT;
  }

  /** The output of this method should be identical to the method "serializeStats(outputStream)". */
  @Override
  public int getStatsSize() {
    return 4 + firstValue.getValues().length + 4 + lastValue.getValues().length;
  }

  @Override
  public long getRetainedSizeInBytes() {
    return INSTANCE_SIZE
        + sizeOfCharArray(firstValue.getLength())
        + sizeOfCharArray(lastValue.getLength());
  }

  /**
   * initialize Statistics.
   *
   * @param first the first value
   * @param last the last value
   */
  public void initializeStats(Binary first, Binary last) {
    this.firstValue = first;
    this.lastValue = last;
  }

  private void updateLastStats(Binary lastValue) {
    this.lastValue = lastValue;
  }

  private void updateStats(Binary firstValue, Binary lastValue, long startTime, long endTime) {
    // only if endTime greater or equals to the current endTime need we update the last value
    // only if startTime less or equals to the current startTime need we update the first value
    // otherwise, just ignore
    if (startTime <= this.getStartTime()) {
      this.firstValue = firstValue;
    }
    if (endTime >= this.getEndTime()) {
      this.lastValue = lastValue;
    }
  }

  @Override
  public Binary getMinValue() {
    throw new StatisticsClassException(
        String.format(STATS_UNSUPPORTED_MSG, TSDataType.TEXT, "min"));
  }

  @Override
  public Binary getMaxValue() {
    throw new StatisticsClassException(
        String.format(STATS_UNSUPPORTED_MSG, TSDataType.TEXT, "max"));
  }

  @Override
  public Binary getFirstValue() {
    return firstValue;
  }

  @Override
  public Binary getLastValue() {
    return lastValue;
  }

  @Override
  public double getSumDoubleValue() {
    throw new StatisticsClassException(
        String.format(STATS_UNSUPPORTED_MSG, TSDataType.TEXT, "double sum"));
  }

  @Override
  public long getSumLongValue() {
    throw new StatisticsClassException(
        String.format(STATS_UNSUPPORTED_MSG, TSDataType.TEXT, "long sum"));
  }

  @SuppressWarnings("rawtypes")
  @Override
  protected void mergeStatisticsValue(Statistics stats) {
    if (stats instanceof BinaryStatistics || stats instanceof StringStatistics) {
      if (isEmpty) {
        initializeStats(((Binary) stats.getFirstValue()), ((Binary) stats.getLastValue()));
        isEmpty = false;
      } else {
        updateStats(
            ((Binary) stats.getFirstValue()),
            ((Binary) stats.getLastValue()),
            stats.getStartTime(),
            stats.getEndTime());
      }
    } else {
      throw new StatisticsClassException(this.getClass(), stats.getClass());
    }
  }

  @Override
  void updateStats(Binary value) {
    if (isEmpty) {
      initializeStats(value, value);
      isEmpty = false;
    } else {
      updateLastStats(value);
    }
  }

  @Override
  void updateStats(Binary[] values, int batchSize) {
    for (int i = 0; i < batchSize; i++) {
      updateStats(values[i]);
    }
  }

  @Override
  public int serializeStats(OutputStream outputStream) throws IOException {
    int byteLen = 0;
    byteLen += ReadWriteIOUtils.write(firstValue, outputStream);
    byteLen += ReadWriteIOUtils.write(lastValue, outputStream);
    return byteLen;
  }

  @Override
  public void deserialize(InputStream inputStream) throws IOException {
    this.firstValue = ReadWriteIOUtils.readBinary(inputStream);
    this.lastValue = ReadWriteIOUtils.readBinary(inputStream);
  }

  @Override
  public void deserialize(ByteBuffer byteBuffer) {
    this.firstValue = ReadWriteIOUtils.readBinary(byteBuffer);
    this.lastValue = ReadWriteIOUtils.readBinary(byteBuffer);
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (o == null || getClass() != o.getClass()) {
      return false;
    }
    if (!super.equals(o)) {
      return false;
    }
    BinaryStatistics that = (BinaryStatistics) o;
    return Objects.equals(firstValue, that.firstValue) && Objects.equals(lastValue, that.lastValue);
  }

  @Override
  public int hashCode() {
    return Objects.hash(super.hashCode(), firstValue, lastValue);
  }

  @Override
  public String toString() {
    return super.toString() + " [firstValue:" + firstValue + ",lastValue:" + lastValue + "]";
  }
}
