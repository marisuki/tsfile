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
package org.apache.tsfile.read;

import org.apache.tsfile.common.conf.TSFileConfig;
import org.apache.tsfile.common.conf.TSFileDescriptor;
import org.apache.tsfile.enums.TSDataType;
import org.apache.tsfile.read.common.Field;
import org.apache.tsfile.read.common.Path;
import org.apache.tsfile.read.common.RowRecord;
import org.apache.tsfile.read.expression.IExpression;
import org.apache.tsfile.read.expression.QueryExpression;
import org.apache.tsfile.read.expression.impl.BinaryExpression;
import org.apache.tsfile.read.expression.impl.GlobalTimeExpression;
import org.apache.tsfile.read.expression.impl.SingleSeriesExpression;
import org.apache.tsfile.read.filter.factory.TimeFilterApi;
import org.apache.tsfile.read.filter.factory.ValueFilterApi;
import org.apache.tsfile.read.query.dataset.QueryDataSet;
import org.apache.tsfile.utils.Binary;
import org.apache.tsfile.utils.FileGenerator;

import org.junit.After;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.apache.tsfile.read.filter.factory.ValueFilterApi.DEFAULT_MEASUREMENT_INDEX;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

public class TimePlainEncodeReadTest {

  private static final String fileName = FileGenerator.outputDataFile;
  private static TsFileReader roTsFile = null;
  private String timeEncoder = null;

  @Before
  public void prepare() throws IOException {
    timeEncoder = TSFileDescriptor.getInstance().getConfig().getTimeEncoder();
    TSFileDescriptor.getInstance().getConfig().setTimeEncoder("PLAIN");
    FileGenerator.generateFile();
    TsFileSequenceReader reader = new TsFileSequenceReader(fileName);
    roTsFile = new TsFileReader(reader);
  }

  @After
  public void after() throws IOException {
    if (roTsFile != null) {
      roTsFile.close();
    }
    FileGenerator.after();
    TSFileDescriptor.getInstance().getConfig().setTimeEncoder(timeEncoder);
  }

  @Test
  public void queryOneMeasurementWithoutFilterTest() throws IOException {
    List<Path> pathList = new ArrayList<>();
    pathList.add(new Path("d1", "s1", true));
    QueryExpression queryExpression = QueryExpression.create(pathList, null);
    QueryDataSet dataSet = roTsFile.query(queryExpression);

    int count = 0;
    while (dataSet.hasNext()) {
      RowRecord r = dataSet.next();
      if (count == 0) {
        assertEquals(1480562618010L, r.getTimestamp());
      }
      if (count == 499) {
        assertEquals(1480562618999L, r.getTimestamp());
      }
      count++;
    }
    assertEquals(500, count);
  }

  @Test
  public void queryTwoMeasurementsWithoutFilterTest() throws IOException {
    List<Path> pathList = new ArrayList<>();
    pathList.add(new Path("d1", "s1", true));
    pathList.add(new Path("d2", "s2", true));
    QueryExpression queryExpression = QueryExpression.create(pathList, null);
    QueryDataSet dataSet = roTsFile.query(queryExpression);

    int count = 0;
    while (dataSet.hasNext()) {
      RowRecord r = dataSet.next();
      if (count == 0) {
        if (count == 0) {
          assertEquals(1480562618005L, r.getTimestamp());
        }
      }
      count++;
    }
    assertEquals(750, count);
  }

  @Test
  public void queryTwoMeasurementsWithSingleFilterTest() throws IOException {
    List<Path> pathList = new ArrayList<>();
    pathList.add(new Path("d2", "s1", true));
    pathList.add(new Path("d2", "s4", true));
    IExpression valFilter =
        new SingleSeriesExpression(
            new Path("d2", "s2", true),
            ValueFilterApi.gt(DEFAULT_MEASUREMENT_INDEX, 9722L, TSDataType.INT64));
    IExpression tFilter =
        BinaryExpression.and(
            new GlobalTimeExpression(TimeFilterApi.gtEq(1480562618970L)),
            new GlobalTimeExpression(TimeFilterApi.lt(1480562618977L)));
    IExpression finalFilter = BinaryExpression.and(valFilter, tFilter);
    QueryExpression queryExpression = QueryExpression.create(pathList, finalFilter);
    QueryDataSet dataSet = roTsFile.query(queryExpression);

    while (dataSet.hasNext()) {
      dataSet.next();
    }
  }

  @Test
  public void queryWithTwoSeriesTimeValueFilterCrossTest() throws IOException {
    List<Path> pathList = new ArrayList<>();
    pathList.add(new Path("d2", "s2", true));
    IExpression valFilter =
        new SingleSeriesExpression(
            new Path("d2", "s2", true),
            ValueFilterApi.notEq(DEFAULT_MEASUREMENT_INDEX, 9722L, TSDataType.INT64));
    IExpression tFilter =
        BinaryExpression.and(
            new GlobalTimeExpression(TimeFilterApi.gtEq(1480562618970L)),
            new GlobalTimeExpression(TimeFilterApi.lt(1480562618977L)));
    IExpression finalFilter = BinaryExpression.and(valFilter, tFilter);
    QueryExpression queryExpression = QueryExpression.create(pathList, finalFilter);
    QueryDataSet dataSet = roTsFile.query(queryExpression);

    // time filter & value filter
    // verify d1.s1, d2.s1
    int cnt = 1;
    while (dataSet.hasNext()) {
      RowRecord r = dataSet.next();
      if (cnt == 1) {
        assertEquals(1480562618970L, r.getTimestamp());
      } else if (cnt == 2) {
        assertEquals(1480562618971L, r.getTimestamp());
      } else if (cnt == 3) {
        assertEquals(1480562618973L, r.getTimestamp());
      }
      cnt++;
    }
    assertEquals(7, cnt);
  }

  @Test
  public void queryWithCrossSeriesTimeValueFilterTest() throws IOException {
    List<Path> pathList = new ArrayList<>();
    pathList.add(new Path("d1", "s1", true));
    pathList.add(new Path("d2", "s2", true));
    IExpression valFilter =
        new SingleSeriesExpression(
            new Path("d2", "s2", true),
            ValueFilterApi.gt(DEFAULT_MEASUREMENT_INDEX, 9722L, TSDataType.INT64));
    IExpression tFilter =
        BinaryExpression.and(
            new GlobalTimeExpression(TimeFilterApi.gtEq(1480562618970L)),
            new GlobalTimeExpression(TimeFilterApi.lt(1480562618977L)));
    IExpression finalFilter = BinaryExpression.and(valFilter, tFilter);
    QueryExpression queryExpression = QueryExpression.create(pathList, finalFilter);
    QueryDataSet dataSet = roTsFile.query(queryExpression);

    // time filter & value filter
    // verify d1.s1, d2.s1
    /**
     * 1480562618950 9501 9502 1480562618954 9541 9542 1480562618955 9551 9552 1480562618956 9561
     * 9562
     */
    int cnt = 1;
    while (dataSet.hasNext()) {
      RowRecord r = dataSet.next();
      if (cnt == 1) {
        assertEquals(1480562618973L, r.getTimestamp());
      } else if (cnt == 2) {
        assertEquals(1480562618974L, r.getTimestamp());
      } else if (cnt == 3) {
        assertEquals(1480562618975L, r.getTimestamp());
      } else if (cnt == 4) {
        assertEquals(1480562618976L, r.getTimestamp());
      }
      cnt++;
    }
    assertEquals(5, cnt);

    pathList.clear();
    pathList.add(new Path("d1", "s1", true));
    pathList.add(new Path("d2", "s2", true));
    valFilter =
        new SingleSeriesExpression(
            new Path("d1", "s1", true),
            ValueFilterApi.ltEq(DEFAULT_MEASUREMENT_INDEX, 9321, TSDataType.INT32));
    valFilter =
        BinaryExpression.and(
            new SingleSeriesExpression(
                new Path("d2", "s2", true),
                ValueFilterApi.ltEq(DEFAULT_MEASUREMENT_INDEX, 9312L, TSDataType.INT64)),
            valFilter);
    tFilter =
        BinaryExpression.and(
            new GlobalTimeExpression(TimeFilterApi.gtEq(1480562618906L)),
            new GlobalTimeExpression(TimeFilterApi.ltEq(1480562618915L)));
    tFilter =
        BinaryExpression.or(
            tFilter,
            BinaryExpression.and(
                new GlobalTimeExpression(TimeFilterApi.gtEq(1480562618928L)),
                new GlobalTimeExpression(TimeFilterApi.ltEq(1480562618933L))));
    finalFilter = BinaryExpression.and(valFilter, tFilter);
    queryExpression = QueryExpression.create(pathList, finalFilter);
    dataSet = roTsFile.query(queryExpression);

    // time filter & value filter
    // verify d1.s1, d2.s1
    /**
     * 1480562618910 9101 9102 1480562618911 9111 9112 1480562618912 9121 9122 1480562618913 9131
     * 9132 1480562618914 9141 9142 1480562618915 9151 9152 1480562618930 9301 9302 1480562618931
     * 9311 9312 1480562618932 9321 9322 1480562618933 9331 9332
     */
    cnt = 1;
    while (dataSet.hasNext()) {
      RowRecord r = dataSet.next();
      if (cnt == 4) {
        assertEquals(1480562618913L, r.getTimestamp());
      } else if (cnt == 7) {
        assertEquals(1480562618930L, r.getTimestamp());
      }
      cnt++;
    }
    assertEquals(9, cnt);
  }

  @Test
  public void queryBooleanTest() throws IOException {
    List<Path> pathList = new ArrayList<>();
    pathList.add(new Path("d1", "s5", true));
    IExpression valFilter =
        new SingleSeriesExpression(
            new Path("d1", "s5", true),
            ValueFilterApi.eq(DEFAULT_MEASUREMENT_INDEX, false, TSDataType.BOOLEAN));
    IExpression tFilter =
        BinaryExpression.and(
            new GlobalTimeExpression(TimeFilterApi.gtEq(1480562618970L)),
            new GlobalTimeExpression(TimeFilterApi.lt(1480562618981L)));
    IExpression finalFilter = BinaryExpression.and(valFilter, tFilter);
    QueryExpression queryExpression = QueryExpression.create(pathList, finalFilter);
    QueryDataSet dataSet = roTsFile.query(queryExpression);

    int cnt = 1;
    while (dataSet.hasNext()) {
      RowRecord r = dataSet.next();
      if (cnt == 1) {
        assertEquals(1480562618972L, r.getTimestamp());
        Field f1 = r.getFields().get(0);
        assertFalse(f1.getBoolV());
      }
      if (cnt == 2) {
        assertEquals(1480562618981L, r.getTimestamp());
        Field f2 = r.getFields().get(0);
        assertFalse(f2.getBoolV());
      }
      cnt++;
    }
  }

  @Test
  public void queryStringTest() throws IOException {
    List<Path> pathList = new ArrayList<>();
    pathList.add(new Path("d1", "s4", true));
    IExpression valFilter =
        new SingleSeriesExpression(
            new Path("d1", "s4", true),
            ValueFilterApi.gt(
                DEFAULT_MEASUREMENT_INDEX,
                new Binary("dog97", TSFileConfig.STRING_CHARSET),
                TSDataType.TEXT));
    IExpression tFilter =
        BinaryExpression.and(
            new GlobalTimeExpression(TimeFilterApi.gtEq(1480562618970L)),
            new GlobalTimeExpression(TimeFilterApi.ltEq(1480562618981L)));
    IExpression finalFilter = BinaryExpression.and(valFilter, tFilter);
    QueryExpression queryExpression = QueryExpression.create(pathList, finalFilter);
    QueryDataSet dataSet = roTsFile.query(queryExpression);

    int cnt = 0;
    while (dataSet.hasNext()) {
      RowRecord r = dataSet.next();
      if (cnt == 0) {
        assertEquals(1480562618976L, r.getTimestamp());
        Field f1 = r.getFields().get(0);
        assertEquals("dog976", f1.toString());
      }
      cnt++;
    }
    Assert.assertEquals(1, cnt);

    pathList = new ArrayList<>();
    pathList.add(new Path("d1", "s4", true));
    valFilter =
        new SingleSeriesExpression(
            new Path("d1", "s4", true),
            ValueFilterApi.lt(
                DEFAULT_MEASUREMENT_INDEX,
                new Binary("dog97", TSFileConfig.STRING_CHARSET),
                TSDataType.TEXT));
    tFilter =
        BinaryExpression.and(
            new GlobalTimeExpression(TimeFilterApi.gtEq(1480562618970L)),
            new GlobalTimeExpression(TimeFilterApi.ltEq(1480562618981L)));
    finalFilter = BinaryExpression.and(valFilter, tFilter);
    queryExpression = QueryExpression.create(pathList, finalFilter);
    dataSet = roTsFile.query(queryExpression);
    cnt = 0;
    while (dataSet.hasNext()) {
      RowRecord r = dataSet.next();
      if (cnt == 1) {
        assertEquals(1480562618976L, r.getTimestamp());
        Field f1 = r.getFields().get(0);
        assertEquals("dog976", f1.getBinaryV().toString());
      }
      cnt++;
    }
    Assert.assertEquals(0, cnt);
  }

  @Test
  public void queryFloatTest() throws IOException {
    List<Path> pathList = new ArrayList<>();
    pathList.add(new Path("d1", "s6", true));
    IExpression valFilter =
        new SingleSeriesExpression(
            new Path("d1", "s6", true),
            ValueFilterApi.gt(DEFAULT_MEASUREMENT_INDEX, 103.0f, TSDataType.FLOAT));
    IExpression tFilter =
        BinaryExpression.and(
            new GlobalTimeExpression(TimeFilterApi.gtEq(1480562618970L)),
            new GlobalTimeExpression(TimeFilterApi.ltEq(1480562618981L)));
    IExpression finalFilter = BinaryExpression.and(valFilter, tFilter);
    QueryExpression queryExpression = QueryExpression.create(pathList, finalFilter);
    QueryDataSet dataSet = roTsFile.query(queryExpression);

    int cnt = 0;
    while (dataSet.hasNext()) {
      RowRecord r = dataSet.next();
      if (cnt == 1) {
        assertEquals(1480562618980L, r.getTimestamp());
        Field f1 = r.getFields().get(0);
        assertEquals(108.0, f1.getFloatV(), 0.0);
      }
      if (cnt == 2) {
        assertEquals(1480562618990L, r.getTimestamp());
        Field f2 = r.getFields().get(0);
        assertEquals(110.0, f2.getFloatV(), 0.0);
      }
      cnt++;
    }
  }

  @Test
  public void queryDoubleTest() throws IOException {
    List<Path> pathList = new ArrayList<>();
    pathList.add(new Path("d1", "s7", true));
    IExpression valFilter =
        new SingleSeriesExpression(
            new Path("d1", "s7", true),
            ValueFilterApi.gt(DEFAULT_MEASUREMENT_INDEX, 7.0, TSDataType.DOUBLE));
    IExpression tFilter =
        BinaryExpression.and(
            new GlobalTimeExpression(TimeFilterApi.gtEq(1480562618021L)),
            new GlobalTimeExpression(TimeFilterApi.ltEq(1480562618033L)));
    IExpression finalFilter = BinaryExpression.and(valFilter, tFilter);
    QueryExpression queryExpression = QueryExpression.create(pathList, finalFilter);
    QueryDataSet dataSet = roTsFile.query(queryExpression);

    int cnt = 1;
    while (dataSet.hasNext()) {
      RowRecord r = dataSet.next();
      if (cnt == 1) {
        assertEquals(1480562618022L, r.getTimestamp());
        Field f1 = r.getFields().get(0);
        assertEquals(2.0, f1.getDoubleV(), 0.0);
      }
      if (cnt == 2) {
        assertEquals(1480562618033L, r.getTimestamp());
        Field f1 = r.getFields().get(0);
        assertEquals(3.0, f1.getDoubleV(), 0.0);
      }
      cnt++;
    }
  }
}
